"""LLM-based mutation and generation of mathematical update rules.

Uses the Anthropic Claude API to generate free-form mathematical expressions
that serve as Lenia-like cellular automaton update rules.

Inspired by the LaSR approach (Language-guided Symbolic Regression):
  - Generate initial diverse expressions
  - Mutate/crossover high-fitness parents
  - Periodically abstract reusable structural patterns
"""

from __future__ import annotations

import os
import re
import time
from typing import Optional

import anthropic


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 2048
TEMPERATURE_GENERATE = 1.0   # High diversity for initial generation
TEMPERATURE_MUTATE = 0.8     # Moderate creativity for mutations
TEMPERATURE_ABSTRACT = 0.5   # More focused for abstraction


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert in mathematical biology and cellular automata, specialising \
in Lenia — a continuous cellular automaton framework for artificial life. \
Your goal is to discover update rules that produce self-organising, persistent, \
moving structures in a 2D grid."""

GENERATE_PROMPT = """\
The state A is a 2D grid of values in [0,1]. Each timestep, A is updated by:
  A_new = <your expression>

Available operations:
  conv(A)        - FFT convolution of A with a Gaussian kernel (radius ~13)
  laplacian(A)   - discrete Laplacian of A
  gradient_x(A)  - horizontal spatial gradient of A
  gradient_y(A)  - vertical spatial gradient of A
  A              - the current state grid
  sin, cos, tanh, exp, abs, sqrt, log, sigmoid, sign
  clip(x, lo, hi) - clamp values to [lo, hi]
  threshold(x, t) - returns 1 where x > t, else 0
  max(a, b), min(a, b) - element-wise max/min
  +, -, *, /, **
  Constants: pi, e, and any number

IMPORTANT CONSTRAINTS:
- The expression MUST return values that stay roughly in [0, 1] (use clip if needed)
- conv(A) returns the local neighbourhood average — values are in [0, 1]
- Use dt values (0.01 to 0.3) to control update speed: A + dt * (...)
- Interesting life requires a balance: growth at some densities, decay at others

Generate {n} DIFFERENT and CREATIVE update rules. Each should be a single-line \
Python expression. Aim for diversity:
  - Some inspired by classic Lenia (Gaussian growth function on convolution)
  - Some using reaction-diffusion patterns (Laplacian + nonlinear growth)
  - Some using novel combinations (gradients, thresholds, multiple convolutions)
  - Some with asymmetric or directional dynamics

Return ONLY the expressions, one per line. No numbering, no explanation, no markdown."""

MUTATE_PROMPT = """\
I am searching for mathematical update rules that produce artificial life — \
self-organising, moving, persistent structures in a 2D grid.

Here are two parent expressions that showed interesting behaviour:

PARENT 1 (fitness={fitness1:.4f}, alive_fraction={alive1:.3f}, complexity={complexity1:.3f}):
  {expr1}

PARENT 2 (fitness={fitness2:.4f}, alive_fraction={alive2:.3f}, complexity={complexity2:.3f}):
  {expr2}

Available operations:
  conv(A), laplacian(A), gradient_x(A), gradient_y(A), A
  sin, cos, tanh, exp, abs, sqrt, log, sigmoid, sign
  clip(x, lo, hi), threshold(x, t), max(a, b), min(a, b)
  +, -, *, /, **, pi, e

Generate ONE new expression that combines structural ideas from both parents. \
The new expression should:
  1. Borrow the most promising mathematical structures from each parent
  2. Introduce a small novel twist (different nonlinearity, extra term, etc.)
  3. Keep values roughly in [0,1] (use clip if needed)
  4. Be a single-line Python expression

Return ONLY the expression, nothing else."""

ABSTRACT_PROMPT = """\
I have been searching for mathematical update rules that produce artificial life. \
Here are the top-performing expressions and their fitness scores:

{expression_list}

Analyse these expressions and identify reusable structural PATTERNS — recurring \
mathematical motifs that appear in high-fitness expressions.

For each pattern, provide:
1. A name for the pattern (e.g., "gaussian_growth_on_convolution")
2. A template expression using PLACEHOLDERS like {{param1}}, {{param2}}
3. What the pattern does conceptually

Then generate 3 NEW expressions that creatively combine these patterns in ways \
not seen in the input expressions.

Format your response as:

PATTERNS:
- pattern_name: template_expression | description

NEW EXPRESSIONS:
expression1
expression2
expression3"""

GENERATE_DUAL_PROMPT = """You are searching for mathematical update rules with TWO coupled fields (A and B) that produce artificial life — self-organizing, moving, persistent structures in a 2D grid.

Each timestep, both fields are updated:
  A_new = <expression using A, B, conv(A), conv(B), laplacian(A), laplacian(B), ...>
  B_new = <expression using A, B, conv(A), conv(B), laplacian(A), laplacian(B), ...>

Available operations:
  Spatial: conv(A), conv(B), laplacian(A), laplacian(B), gradient_x(A), gradient_y(A), gradient_x(B), gradient_y(B)
  High-level: conserve_a(A, B) (normalize so A+B=1), conserve_b(A, B), compete(A, B) (A suppressed where B strong), curl(A) (vorticity), rotate_cw(A) (clockwise chirality), rotate_ccw(A), global_mean(A)
  Math: sin, cos, tanh, exp, abs, sqrt, clip(x, 0, 1), threshold(x, t), +, -, *, /, **, pi

Here are CONCRETE TEMPLATES for different coupling strategies. Use these as starting points and modify them:

CONSERVATION (matter transforms between fields):
  A: clip(A + 0.1 * (2*exp(-((conv(A)-0.15)/0.02)**2) - 1) - 0.05 * A * B, 0, 1)
  B: conserve_b(A + 0.1 * (2*exp(-((conv(A)-0.15)/0.02)**2) - 1) - 0.05*A*B, B + 0.05*A*B)

COMPETITION (two species fight for territory):
  A: clip(A + 0.1 * (2*exp(-((conv(A)-0.15)/0.03)**2) - 1) * compete(A, B), 0, 1)
  B: clip(B + 0.1 * (2*exp(-((conv(B)-0.2)/0.03)**2) - 1) * compete(B, A), 0, 1)

CHIRAL (left-right asymmetry, spinning structures):
  A: clip(A + 0.1 * (2*exp(-((conv(A)-0.15)/0.02)**2) - 1) + 0.02 * curl(A) * B, 0, 1)
  B: clip(B + 0.05 * laplacian(B) + 0.03 * rotate_cw(A) - 0.02 * B, 0, 1)

PREDATOR-PREY (A eats B, B regrows):
  A: clip(A + 0.1 * A * conv(B) - 0.05 * A + 0.01 * laplacian(A), 0, 1)
  B: clip(B + 0.15 * B * (1 - B) - 0.12 * A * B + 0.02 * laplacian(B), 0, 1)

WAVE-PARTICLE (A = continuous wave, B = discrete blobs):
  A: clip(A + 0.1 * laplacian(A) + 0.05 * sin(4*pi*conv(A)) * threshold(B, 0.3), 0, 1)
  B: clip(B + 0.1 * (2*exp(-((conv(B)-0.2)/0.03)**2) - 1) * (1 + 0.3*A), 0, 1)

IMPORTANT: Do NOT just copy these templates. Use them as inspiration, then modify coefficients, add extra terms, combine ideas from different templates, or invent entirely new coupling structures.

Generate {n} different DUAL update rule pairs. Format each pair as exactly two lines:
A: <expression for A>
B: <expression for B>

Separate pairs with a blank line. No explanation, just the expressions."""

MUTATE_DUAL_PROMPT = """I am searching for mathematical update rules with TWO coupled fields (A and B) that produce artificial life — self-organising, moving, persistent structures in a 2D grid.

Here are two parent expression pairs that showed interesting behaviour:

PARENT 1 (fitness={fitness1:.4f}):
  A: {expr_a1}
  B: {expr_b1}

PARENT 2 (fitness={fitness2:.4f}):
  A: {expr_a2}
  B: {expr_b2}

Available operations:
  Spatial: conv(A), conv(B), laplacian(A), laplacian(B), gradient_x(A), gradient_y(A), gradient_x(B), gradient_y(B)
  High-level: conserve_a(A, B), conserve_b(A, B), compete(A, B), curl(A), rotate_cw(A), rotate_ccw(A), global_mean(A)
  Math: sin, cos, tanh, exp, abs, sqrt, clip(x, 0, 1), threshold(x, t), +, -, *, /, **, pi

Generate ONE new expression pair. The new pair should:
  1. Borrow the most promising mathematical structures from each parent
  2. Try using HIGH-LEVEL operations the parents didn't use: conserve_a/b (mass conservation), compete (competition), curl/rotate_cw/rotate_ccw (chirality), global_mean (global coupling)
  3. Keep values roughly in [0,1] (use clip if needed)
  4. Aim for behavior DIFFERENT from both parents — try a fundamentally different coupling strategy, not just tweaked coefficients

Return ONLY two lines:
A: <expression for A>
B: <expression for B>"""




# ---------------------------------------------------------------------------
# LLMMutator class
# ---------------------------------------------------------------------------

class LLMMutator:
    """Uses Claude API to generate and mutate Lenia update rule expressions."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """Initialise the mutator.

        Parameters
        ----------
        api_key : str, optional
            Anthropic API key. If None, uses ANTHROPIC_API_KEY env var.
        base_url : str, optional
            Custom API base URL (for aggregation services). If None, uses
            ANTHROPIC_BASE_URL env var or Anthropic default.
        """
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        elif os.environ.get("ANTHROPIC_BASE_URL"):
            kwargs["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
        self.client = anthropic.Anthropic(**kwargs) if kwargs else anthropic.Anthropic()

        self.discovered_patterns: list[dict] = []
        self._call_count = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def _call_api(
        self,
        user_prompt: str,
        temperature: float = 0.8,
        max_tokens: int = MAX_TOKENS,
    ) -> str:
        """Make a single API call with retry logic."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=MODEL,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                self._call_count += 1
                self._total_input_tokens += response.usage.input_tokens
                self._total_output_tokens += response.usage.output_tokens
                return response.content[0].text
            except anthropic.RateLimitError:
                wait = 2 ** attempt * 5
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
            except anthropic.APIError as exc:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt * 2
                    print(f"  API error: {exc}, retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Failed after all retries")

    @staticmethod
    def _parse_expressions(text: str) -> list[str]:
        """Extract valid-looking expressions from API response text.

        Filters out blank lines, comments, numbered prefixes, markdown, etc.
        """
        lines = text.strip().split("\n")
        expressions = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip markdown code fences
            if line.startswith("```"):
                continue
            # Strip numbering like "1. " or "1) " or "- "
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^[-*]\s*", "", line)
            line = line.strip()
            # Skip lines that look like comments or descriptions
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            # Must contain at least 'A' (the state variable) to be a valid expression
            if "A" not in line:
                continue
            # Skip very long lines (likely descriptions, not expressions)
            if len(line) > 500:
                continue
            expressions.append(line)
        return expressions

    def generate_initial(self, n: int = 5) -> list[str]:
        """Generate n initial random update rule expressions.

        Parameters
        ----------
        n : int
            Number of expressions to generate.

        Returns
        -------
        list[str]
            List of expression strings.
        """
        prompt = GENERATE_PROMPT.format(n=n)
        response = self._call_api(prompt, temperature=TEMPERATURE_GENERATE)
        expressions = self._parse_expressions(response)

        # If we got fewer than requested, make another call
        if len(expressions) < n:
            prompt2 = GENERATE_PROMPT.format(n=n - len(expressions))
            response2 = self._call_api(prompt2, temperature=TEMPERATURE_GENERATE)
            expressions.extend(self._parse_expressions(response2))

        return expressions[:n]

    def mutate(self, parent1: dict, parent2: dict) -> str:
        """Given two high-fitness parents, generate a new expression.

        Parameters
        ----------
        parent1 : dict
            Must have keys: "expression", "fitness", "metrics" (with alive_fraction, complexity).
        parent2 : dict
            Same format as parent1.

        Returns
        -------
        str
            A new expression string.
        """
        # Extract info from parents
        expr1 = parent1.get("expression", parent1.get("genome", {}).get("expression", "A"))
        expr2 = parent2.get("expression", parent2.get("genome", {}).get("expression", "A"))
        metrics1 = parent1.get("metrics", {})
        metrics2 = parent2.get("metrics", {})

        prompt = MUTATE_PROMPT.format(
            expr1=expr1,
            expr2=expr2,
            fitness1=parent1.get("fitness", 0.0),
            fitness2=parent2.get("fitness", 0.0),
            alive1=metrics1.get("alive_fraction", 0.0),
            alive2=metrics2.get("alive_fraction", 0.0),
            complexity1=metrics1.get("complexity", 0.0),
            complexity2=metrics2.get("complexity", 0.0),
        )

        # Include discovered patterns if available
        if self.discovered_patterns:
            pattern_text = "\n\nKnown useful patterns:\n"
            for p in self.discovered_patterns[-5:]:  # Last 5 patterns
                pattern_text += f"  - {p['name']}: {p['template']}\n"
            prompt += pattern_text

        response = self._call_api(prompt, temperature=TEMPERATURE_MUTATE)
        expressions = self._parse_expressions(response)

        if not expressions:
            # Fallback: return a simple mutation of parent1
            return expr1

        return expressions[0]

    def abstract_concepts(self, top_expressions: list[dict]) -> list[dict]:
        """LaSR-style: extract reusable structural patterns from top expressions.

        Parameters
        ----------
        top_expressions : list[dict]
            Each dict has "expression" and "fitness" keys.

        Returns
        -------
        list[dict]
            New expressions generated from abstracted patterns.
            Each dict has "expression" and "source" keys.
        """
        if len(top_expressions) < 3:
            return []

        # Build the expression list for the prompt
        expr_lines = []
        for i, entry in enumerate(top_expressions[:15], 1):
            expr = entry.get("expression", entry.get("genome", {}).get("expression", "?"))
            fitness = entry.get("fitness", 0.0)
            expr_lines.append(f"  {i}. (fitness={fitness:.4f}) {expr}")
        expression_list = "\n".join(expr_lines)

        prompt = ABSTRACT_PROMPT.format(expression_list=expression_list)
        response = self._call_api(prompt, temperature=TEMPERATURE_ABSTRACT)

        # Parse patterns
        new_patterns = []
        if "PATTERNS:" in response:
            pattern_section = response.split("PATTERNS:")[1]
            if "NEW EXPRESSIONS:" in pattern_section:
                pattern_section = pattern_section.split("NEW EXPRESSIONS:")[0]
            for line in pattern_section.strip().split("\n"):
                line = line.strip()
                if line.startswith("- ") and ":" in line:
                    parts = line[2:].split(":", 1)
                    name = parts[0].strip()
                    rest = parts[1].strip() if len(parts) > 1 else ""
                    template = rest.split("|")[0].strip() if "|" in rest else rest
                    new_patterns.append({"name": name, "template": template})

        if new_patterns:
            self.discovered_patterns.extend(new_patterns)
            # Keep only the most recent 20 patterns
            self.discovered_patterns = self.discovered_patterns[-20:]

        # Parse new expressions
        new_expressions = []
        if "NEW EXPRESSIONS:" in response:
            expr_section = response.split("NEW EXPRESSIONS:")[1]
            parsed = self._parse_expressions(expr_section)
            for expr in parsed:
                new_expressions.append({
                    "expression": expr,
                    "source": "abstraction",
                })

        return new_expressions


    @staticmethod
    def _parse_dual_expressions(text: str) -> list[tuple[str, str]]:
        """Extract dual (A, B) expression pairs from API response text.

        Expects pairs formatted as:
            A: <expression>
            B: <expression>

        Returns list of (expr_a, expr_b) tuples.
        """
        lines = text.strip().split("\n")
        pairs = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            # Skip blank lines, markdown fences, comments
            if not line or line.startswith("```") or line.startswith("#"):
                i += 1
                continue
            # Look for "A:" prefix
            expr_a = None
            expr_b = None
            if line.upper().startswith("A:"):
                expr_a = line[2:].strip()
                # Strip numbering/bullets from the expression
                expr_a = re.sub(r"^\d+[\.)\]]\s*", "", expr_a).strip()
                # Look for B: on next non-blank line
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines):
                    bline = lines[j].strip()
                    if bline.upper().startswith("B:"):
                        expr_b = bline[2:].strip()
                        expr_b = re.sub(r"^\d+[\.)\]]\s*", "", expr_b).strip()
                        i = j + 1
                    else:
                        i += 1
                else:
                    i += 1
            else:
                i += 1
                continue

            # Validate: both expressions must exist and reference expected variables
            if expr_a and expr_b and "A" in expr_a and len(expr_a) < 500 and len(expr_b) < 500:
                pairs.append((expr_a, expr_b))

        return pairs

    def generate_dual_initial(self, n: int = 5) -> list[tuple[str, str]]:
        """Generate n initial dual-field update rule expression pairs.

        Parameters
        ----------
        n : int
            Number of expression pairs to generate.

        Returns
        -------
        list[tuple[str, str]]
            List of (expr_a, expr_b) tuples.
        """
        prompt = GENERATE_DUAL_PROMPT.format(n=n)
        response = self._call_api(prompt, temperature=TEMPERATURE_GENERATE)
        pairs = self._parse_dual_expressions(response)

        # If we got fewer than requested, make another call
        if len(pairs) < n:
            prompt2 = GENERATE_DUAL_PROMPT.format(n=n - len(pairs))
            response2 = self._call_api(prompt2, temperature=TEMPERATURE_GENERATE)
            pairs.extend(self._parse_dual_expressions(response2))

        return pairs[:n]

    def mutate_dual(
        self, parent1: dict, parent2: dict
    ) -> tuple[str, str]:
        """Given two high-fitness dual-field parents, generate a new expression pair.

        Parameters
        ----------
        parent1 : dict
            Must have keys: "expression_a", "expression_b", "fitness".
            Or "genome" with those keys.
        parent2 : dict
            Same format as parent1.

        Returns
        -------
        tuple[str, str]
            (expr_a, expr_b) for the new dual-field update rule.
        """
        genome1 = parent1.get("genome", parent1)
        genome2 = parent2.get("genome", parent2)

        expr_a1 = genome1.get("expression_a", genome1.get("expression", "A"))
        expr_b1 = genome1.get("expression_b", "B")
        expr_a2 = genome2.get("expression_a", genome2.get("expression", "A"))
        expr_b2 = genome2.get("expression_b", "B")

        prompt = MUTATE_DUAL_PROMPT.format(
            expr_a1=expr_a1,
            expr_b1=expr_b1,
            expr_a2=expr_a2,
            expr_b2=expr_b2,
            fitness1=parent1.get("fitness", 0.0),
            fitness2=parent2.get("fitness", 0.0),
        )

        # Include discovered patterns if available
        if self.discovered_patterns:
            pattern_text = "\n\nKnown useful patterns:\n"
            for p in self.discovered_patterns[-5:]:
                pattern_text += f"  - {p['name']}: {p['template']}\n"
            prompt += pattern_text

        response = self._call_api(prompt, temperature=TEMPERATURE_MUTATE)
        pairs = self._parse_dual_expressions(response)

        if not pairs:
            # Fallback: return parent1's expressions
            return (expr_a1, expr_b1)

        return pairs[0]

    def get_stats(self) -> dict:
        """Return API usage statistics."""
        return {
            "api_calls": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "discovered_patterns": len(self.discovered_patterns),
        }
