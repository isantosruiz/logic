from __future__ import annotations

import random
import re
from dataclasses import dataclass
from itertools import product
from typing import Any

from flask import Flask, jsonify, render_template_string, request
from sympy import Symbol, false, latex, symbols, true
from sympy.core.sympify import SympifyError
from sympy.logic.boolalg import And, Not, Or, POSform, SOPform, Xor, simplify_logic
from sympy.parsing.sympy_parser import parse_expr

app = Flask(__name__)

RESULT_FORMS = {"sop", "pos"}
INPUT_MODES = {"equation", "table", "minterms", "maxterms"}
EQUIVALENCE_METHODS = {"auto", "exhaustive", "sample"}
CONVERT_DIRECTIONS = {"equation_to_table", "table_to_equation"}

VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
WORD_OPERATORS = {
    r"\bAND\b": "&",
    r"\bOR\b": "|",
    r"\bNOT\b": "~",
    r"\bXOR\b": "^",
    r"\bTRUE\b": "True",
    r"\bFALSE\b": "False",
}

DEFAULT_SAMPLE_SIZE = 256
MAX_SAMPLE_SIZE = 4096
MAX_EXHAUSTIVE_VARIABLES = 12


@dataclass
class ParsedTable:
    variables: list[str]
    minterms: list[int]
    maxterms: list[int]
    dontcares: list[int]


def _split_tokens(raw: str) -> list[str]:
    return [token for token in re.split(r"[,;\s]+", raw.strip()) if token]


def parse_variables(raw: Any) -> list[str]:
    if raw is None:
        return []

    if isinstance(raw, list):
        tokens = [str(item).strip() for item in raw if str(item).strip()]
    else:
        tokens = _split_tokens(str(raw))

    variables: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not VAR_NAME_RE.match(token):
            raise ValueError(
                f"Nombre de variable inválido: '{token}'. Use letras, números y '_' (sin comenzar con número)."
            )
        if token not in seen:
            variables.append(token)
            seen.add(token)

    return variables


def parse_index_list(raw: Any) -> list[int]:
    if raw is None:
        return []

    text = str(raw).strip()
    if not text:
        return []

    values: list[int] = []
    seen: set[int] = set()
    for token in _split_tokens(text):
        try:
            value = int(token)
        except ValueError as exc:
            raise ValueError(f"Índice inválido: '{token}'.") from exc

        if value < 0:
            raise ValueError("Los índices deben ser enteros >= 0.")

        if value not in seen:
            values.append(value)
            seen.add(value)

    return sorted(values)


def make_boolean_symbols(variable_names: list[str]) -> tuple[Symbol, ...]:
    if not variable_names:
        return tuple()

    generated = symbols(" ".join(variable_names), boolean=True)
    if isinstance(generated, tuple):
        return generated
    return (generated,)


def simplify_boolean_form(expression: Any, form: str):
    # With an explicit form, SymPy can leave an already-normal expression such
    # as A | ~A untouched. The form-free pass detects constants reliably.
    form_free = simplify_logic(expression, force=True)
    if form_free in {true, false}:
        return form_free

    return simplify_logic(expression, form=form, force=True)


def _normalize_equation(raw: str) -> str:
    equation = raw.strip()
    if "=" in equation:
        _, equation = equation.split("=", 1)

    equation = equation.strip()

    replacements = {
        "¬": "~",
        "!": "~",
        "·": "&",
        "*": "&",
        "+": "|",
        "∧": "&",
        "∨": "|",
    }
    for source, target in replacements.items():
        equation = equation.replace(source, target)

    for pattern, replacement in WORD_OPERATORS.items():
        equation = re.sub(pattern, replacement, equation, flags=re.IGNORECASE)

    return equation


def parse_equation(raw: str, explicit_variables: list[str]) -> tuple[Any, list[str]]:
    normalized = _normalize_equation(raw)
    if not normalized:
        raise ValueError("Debe ingresar una ecuación lógica.")

    discovered = [
        token
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", normalized)
        if token.lower() not in {"and", "or", "not", "xor", "true", "false"}
    ]

    ordered_discovered: list[str] = []
    for token in discovered:
        if token not in ordered_discovered:
            ordered_discovered.append(token)

    variable_names = explicit_variables or ordered_discovered
    local_symbols = {name: Symbol(name, boolean=True) for name in variable_names}

    try:
        expression = parse_expr(normalized, local_dict=local_symbols, evaluate=False)
    except (SympifyError, SyntaxError) as exc:
        raise ValueError(
            "No se pudo interpretar la ecuación. Use operadores como ~, &, | y ^."
        ) from exc

    if expression == True:
        expression = true
    elif expression == False:
        expression = false

    free_symbols = {str(item) for item in getattr(expression, "free_symbols", set())}
    unknown = free_symbols.difference(variable_names)
    if unknown:
        missing = ", ".join(sorted(unknown))
        raise ValueError(f"Variables no declaradas en la lista: {missing}.")

    if explicit_variables:
        return expression, explicit_variables

    inferred = [name for name in ordered_discovered if name in free_symbols]
    return expression, inferred


def _parse_table_line(line: str) -> list[str]:
    if "," in line:
        return [token.strip() for token in line.split(",") if token.strip()]
    return [token.strip() for token in line.split() if token.strip()]


def parse_table(raw: str) -> ParsedTable:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("La tabla debe tener al menos dos filas.")

    first_row = _parse_table_line(lines[0])
    data_lines: list[str]

    if all(token in {"0", "1"} for token in first_row):
        if len(first_row) < 2:
            raise ValueError("Cada fila debe incluir entradas y salida.")
        variables = [f"x{i + 1}" for i in range(len(first_row) - 1)]
        data_lines = lines
    else:
        header = first_row
        if len(header) < 2:
            raise ValueError("El encabezado debe incluir variables y la salida.")
        variables = header[:-1]
        for name in variables:
            if not VAR_NAME_RE.match(name):
                raise ValueError(f"Variable inválida en encabezado: '{name}'.")
        if len(set(variables)) != len(variables):
            raise ValueError("Las variables de la tabla no pueden repetirse.")
        data_lines = lines[1:]

    n_variables = len(variables)
    if n_variables == 0:
        raise ValueError("Se requiere al menos una variable para procesar la tabla.")

    assignments: dict[int, int | str] = {}
    for raw_line in data_lines:
        row = _parse_table_line(raw_line)
        if len(row) != n_variables + 1:
            raise ValueError(
                f"Fila inválida: '{raw_line}'. Debe tener {n_variables + 1} columnas."
            )

        input_bits = row[:-1]
        if any(bit not in {"0", "1"} for bit in input_bits):
            raise ValueError(
                f"Fila inválida: '{raw_line}'. Las entradas deben ser 0 o 1."
            )

        index = int("".join(input_bits), 2)

        output_token = row[-1].lower()
        if output_token in {"1", "true", "t"}:
            output_value: int | str = 1
        elif output_token in {"0", "false", "f"}:
            output_value = 0
        elif output_token in {"x", "-", "dc", "d", "?"}:
            output_value = "x"
        else:
            raise ValueError(
                f"Salida inválida en fila '{raw_line}'. Use 0, 1 o x (don't care)."
            )

        if index in assignments and assignments[index] != output_value:
            raise ValueError(
                f"La combinación {''.join(input_bits)} está repetida con salidas distintas."
            )
        assignments[index] = output_value

    universe_size = 2**n_variables
    dontcares = sorted(
        [index for index, value in assignments.items() if value == "x"]
        + [index for index in range(universe_size) if index not in assignments]
    )
    minterms = sorted(index for index, value in assignments.items() if value == 1)
    maxterms = sorted(index for index, value in assignments.items() if value == 0)

    return ParsedTable(
        variables=variables,
        minterms=minterms,
        maxterms=maxterms,
        dontcares=sorted(set(dontcares)),
    )


def bits_to_index(bits: tuple[int, ...]) -> int:
    if not bits:
        return 0
    return int("".join(str(bit) for bit in bits), 2)


def index_to_bits(index: int, width: int) -> tuple[int, ...]:
    if width == 0:
        return tuple()
    return tuple(int(char) for char in format(index, f"0{width}b"))


def validate_indices(
    variable_names: list[str],
    minterms: list[int],
    dontcares: list[int],
    maxterms: list[int] | None = None,
) -> None:
    universe_size = 2 ** len(variable_names)

    invalid = [value for value in minterms + dontcares if value >= universe_size]
    if invalid:
        raise ValueError(
            f"Índices fuera de rango para {len(variable_names)} variables (0 a {universe_size - 1})."
        )

    if set(minterms).intersection(dontcares):
        raise ValueError("Un índice no puede estar en minitérminos y don't cares a la vez.")

    if maxterms is not None:
        invalid_max = [value for value in maxterms if value >= universe_size]
        if invalid_max:
            raise ValueError(
                f"Maxitérminos fuera de rango para {len(variable_names)} variables (0 a {universe_size - 1})."
            )
        if set(maxterms).intersection(dontcares):
            raise ValueError("Un índice no puede estar en maxitérminos y don't cares a la vez.")


def simplify_from_minterms(
    variable_names: list[str],
    minterms: list[int],
    dontcares: list[int],
    result_form: str,
):
    validate_indices(variable_names, minterms, dontcares)

    if not variable_names:
        return true if minterms else false

    bool_symbols = make_boolean_symbols(variable_names)
    if result_form == "sop":
        expression = SOPform(bool_symbols, minterms, dontcares)
        return simplify_boolean_form(expression, "dnf")

    expression = POSform(bool_symbols, minterms, dontcares)
    return simplify_boolean_form(expression, "cnf")


def minterms_from_maxterms(
    variable_names: list[str],
    maxterms: list[int],
    dontcares: list[int],
) -> list[int]:
    validate_indices(variable_names, [], dontcares, maxterms=maxterms)
    universe = set(range(2 ** len(variable_names)))
    return sorted(universe.difference(maxterms).difference(dontcares))


def summarize_terms(
    variable_names: list[str],
    minterms: list[int],
    dontcares: list[int],
) -> dict[str, list[int]]:
    universe = set(range(2 ** len(variable_names)))
    maxterms = sorted(universe.difference(minterms).difference(dontcares))
    return {
        "minterms": sorted(minterms),
        "maxterms": maxterms,
        "dontcares": sorted(dontcares),
    }


def derive_terms_from_expression(expression: Any, variable_names: list[str]) -> dict[str, list[int]]:
    if not variable_names:
        return {
            "minterms": [0] if bool(expression) else [],
            "maxterms": [] if bool(expression) else [0],
            "dontcares": [],
        }

    bool_symbols = make_boolean_symbols(variable_names)
    minterms: list[int] = []
    for index, bits in enumerate(product([0, 1], repeat=len(bool_symbols))):
        substitution = {symbol: bit for symbol, bit in zip(bool_symbols, bits)}
        if bool(expression.subs(substitution)):
            minterms.append(index)

    universe = set(range(2 ** len(variable_names)))
    maxterms = sorted(universe.difference(minterms))

    return {
        "minterms": minterms,
        "maxterms": maxterms,
        "dontcares": [],
    }


def build_truth_table_from_terms(
    variable_names: list[str], minterms: list[int], dontcares: list[int]
) -> dict[str, Any]:
    n_variables = len(variable_names)
    minterm_set = set(minterms)
    dontcare_set = set(dontcares)

    rows: list[dict[str, Any]] = []
    for index in range(2**n_variables):
        bits = index_to_bits(index, n_variables)
        if index in dontcare_set:
            output = "x"
        elif index in minterm_set:
            output = "1"
        else:
            output = "0"
        rows.append({"index": index, "bits": list(bits), "output": output})

    csv_lines = [",".join(variable_names + ["F"])]
    for row in rows:
        csv_lines.append(",".join([*(str(bit) for bit in row["bits"]), row["output"]]))

    return {
        "header": variable_names + ["F"],
        "rows": rows,
        "csv": "\n".join(csv_lines),
    }


def build_truth_table_from_expression(expression: Any, variable_names: list[str]) -> dict[str, Any]:
    if not variable_names:
        output = "1" if bool(expression) else "0"
        csv_value = "F\n" + output
        return {
            "header": ["F"],
            "rows": [{"index": 0, "bits": [], "output": output}],
            "csv": csv_value,
        }

    bool_symbols = make_boolean_symbols(variable_names)
    rows: list[dict[str, Any]] = []

    for index in range(2 ** len(variable_names)):
        bits = index_to_bits(index, len(variable_names))
        substitution = {symbol: bit for symbol, bit in zip(bool_symbols, bits)}
        value = "1" if bool(expression.subs(substitution)) else "0"
        rows.append({"index": index, "bits": list(bits), "output": value})

    csv_lines = [",".join(variable_names + ["F"])]
    for row in rows:
        csv_lines.append(",".join([*(str(bit) for bit in row["bits"]), row["output"]]))

    return {
        "header": variable_names + ["F"],
        "rows": rows,
        "csv": "\n".join(csv_lines),
    }


def _generate_sample_indices(
    universe_size: int,
    ignored_indices: set[int],
    sample_size: int,
    rng: random.Random,
) -> list[int]:
    available = universe_size - len([index for index in ignored_indices if 0 <= index < universe_size])
    if available <= 0:
        return []

    k = min(sample_size, available)

    if universe_size <= 8192:
        population = [index for index in range(universe_size) if index not in ignored_indices]
        if k >= len(population):
            return population
        return sorted(rng.sample(population, k))

    sampled: set[int] = set()
    max_attempts = k * 40 + 100
    attempts = 0
    while len(sampled) < k and attempts < max_attempts:
        candidate = rng.randrange(universe_size)
        if candidate not in ignored_indices:
            sampled.add(candidate)
        attempts += 1

    if len(sampled) < k:
        for index in range(universe_size):
            if index in ignored_indices:
                continue
            sampled.add(index)
            if len(sampled) >= k:
                break

    return sorted(sampled)


def compare_expressions(
    expression_a: Any,
    expression_b: Any,
    variable_names: list[str],
    method: str = "auto",
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    ignored_indices: list[int] | None = None,
) -> dict[str, Any]:
    if method not in EQUIVALENCE_METHODS:
        raise ValueError("Método inválido para equivalencia. Use auto, exhaustive o sample.")

    if sample_size <= 0:
        raise ValueError("El tamaño de muestra debe ser un entero > 0.")

    effective_sample_size = min(sample_size, MAX_SAMPLE_SIZE)
    ignored = set(ignored_indices or [])

    if not variable_names:
        if 0 in ignored:
            return {
                "equivalent": True,
                "method": "exhaustive",
                "checked_cases": 0,
                "total_care_cases": 0,
                "guaranteed": True,
                "counterexample": None,
            }

        value_a = 1 if bool(expression_a) else 0
        value_b = 1 if bool(expression_b) else 0
        equivalent = value_a == value_b
        return {
            "equivalent": equivalent,
            "method": "exhaustive",
            "checked_cases": 1,
            "total_care_cases": 1,
            "guaranteed": True,
            "counterexample": None
            if equivalent
            else {
                "index": 0,
                "assignment": {},
                "value_a": value_a,
                "value_b": value_b,
            },
        }

    n_variables = len(variable_names)
    universe_size = 2**n_variables
    valid_ignored = {index for index in ignored if 0 <= index < universe_size}
    total_care_cases = universe_size - len(valid_ignored)

    if method == "auto":
        method_used = "exhaustive" if n_variables <= MAX_EXHAUSTIVE_VARIABLES else "sample"
    else:
        method_used = method

    if method_used == "exhaustive" and n_variables > MAX_EXHAUSTIVE_VARIABLES:
        raise ValueError(
            f"El método exhaustivo está limitado a {MAX_EXHAUSTIVE_VARIABLES} variables en esta app."
        )

    if method_used == "exhaustive":
        indices = [index for index in range(universe_size) if index not in valid_ignored]
        guaranteed = True
    else:
        rng = random.Random(42)
        indices = _generate_sample_indices(
            universe_size, valid_ignored, effective_sample_size, rng
        )
        guaranteed = False

    bool_symbols = make_boolean_symbols(variable_names)
    checked_cases = 0
    for index in indices:
        bits = index_to_bits(index, n_variables)
        substitution = {symbol: bit for symbol, bit in zip(bool_symbols, bits)}
        value_a = 1 if bool(expression_a.subs(substitution)) else 0
        value_b = 1 if bool(expression_b.subs(substitution)) else 0
        checked_cases += 1

        if value_a != value_b:
            return {
                "equivalent": False,
                "method": method_used,
                "checked_cases": checked_cases,
                "total_care_cases": total_care_cases,
                "guaranteed": guaranteed,
                "counterexample": {
                    "index": index,
                    "assignment": {
                        variable: bit for variable, bit in zip(variable_names, bits)
                    },
                    "value_a": value_a,
                    "value_b": value_b,
                },
            }

    return {
        "equivalent": True,
        "method": method_used,
        "checked_cases": checked_cases,
        "total_care_cases": total_care_cases,
        "guaranteed": guaranteed,
        "counterexample": None,
    }


def verify_expression_against_terms(
    expression: Any,
    variable_names: list[str],
    minterms: list[int],
    dontcares: list[int],
) -> dict[str, Any]:
    if not variable_names:
        expected = true if minterms else false
    else:
        bool_symbols = make_boolean_symbols(variable_names)
        expected = SOPform(bool_symbols, minterms, dontcares)

    return compare_expressions(
        expected,
        expression,
        variable_names,
        method="auto",
        sample_size=DEFAULT_SAMPLE_SIZE,
        ignored_indices=dontcares,
    )


def _accumulate_gate_counters(target: dict[str, int], source: dict[str, int]) -> None:
    for key in target:
        target[key] += source.get(key, 0)


def _parse_boolean_literal(node: Any) -> tuple[str, bool] | None:
    if isinstance(node, Symbol):
        return str(node), False

    if getattr(node, "func", None) is Not and len(node.args) == 1:
        arg = node.args[0]
        if isinstance(arg, Symbol):
            return str(arg), True

    return None


def _extract_dnf_terms(expression: Any) -> tuple[Any, list[list[tuple[str, bool]]]]:
    dnf = simplify_boolean_form(expression, "dnf")
    if dnf in {true, false}:
        return dnf, []

    term_nodes = dnf.args if getattr(dnf, "func", None) is Or else (dnf,)
    terms: list[list[tuple[str, bool]]] = []

    for term_node in term_nodes:
        literal_nodes = (
            term_node.args if getattr(term_node, "func", None) is And else (term_node,)
        )
        literals: list[tuple[str, bool]] = []
        for literal_node in literal_nodes:
            parsed = _parse_boolean_literal(literal_node)
            if parsed is None:
                raise ValueError(
                    f"No se pudo convertir el literal '{literal_node}' a red NAND."
                )
            literals.append(parsed)
        terms.append(literals)

    return dnf, terms


def _extract_cnf_clauses(expression: Any) -> tuple[Any, list[list[tuple[str, bool]]]]:
    cnf = simplify_boolean_form(expression, "cnf")
    if cnf in {true, false}:
        return cnf, []

    clause_nodes = cnf.args if getattr(cnf, "func", None) is And else (cnf,)
    clauses: list[list[tuple[str, bool]]] = []

    for clause_node in clause_nodes:
        literal_nodes = (
            clause_node.args
            if getattr(clause_node, "func", None) is Or
            else (clause_node,)
        )
        literals: list[tuple[str, bool]] = []
        for literal_node in literal_nodes:
            parsed = _parse_boolean_literal(literal_node)
            if parsed is None:
                raise ValueError(
                    f"No se pudo convertir el literal '{literal_node}' a red NOR."
                )
            literals.append(parsed)
        clauses.append(literals)

    return cnf, clauses


def _add_gate(
    gates: list[dict[str, Any]],
    gate_type: str,
    inputs: list[str],
    comment: str | None = None,
) -> str:
    output_name = f"g{len(gates) + 1}"
    gate = {
        "name": output_name,
        "type": gate_type,
        "inputs": inputs,
        "definition": f"{output_name} = {gate_type}({', '.join(inputs)})",
    }
    if comment:
        gate["comment"] = comment
    gates.append(gate)
    return output_name


def _build_network_payload(
    gate_type: str,
    gates: list[dict[str, Any]],
    output_signal: str,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    netlist_lines: list[str] = []
    for gate in gates:
        line = gate["definition"]
        if gate.get("comment"):
            line += f"  // {gate['comment']}"
        netlist_lines.append(line)
    netlist_lines.append(f"F = {output_signal}")

    return {
        "gate_type": gate_type,
        "gate_count": len(gates),
        "output_signal": output_signal,
        "gates": gates,
        "netlist": "\n".join(netlist_lines),
        "notes": notes or [],
    }


def _literal_text(literal: tuple[str, bool]) -> str:
    name, negated = literal
    return f"~{name}" if negated else name


def _make_constant_nand_network(value: bool, variable_names: list[str]) -> dict[str, Any]:
    if not variable_names:
        return _build_network_payload(
            "NAND",
            [],
            "1" if value else "0",
            notes=[
                "Función constante sin variables: en hardware real se requiere una fuente fija de 1/0.",
            ],
        )

    seed = variable_names[0]
    gates: list[dict[str, Any]] = []
    not_seed = _add_gate(gates, "NAND", [seed, seed], comment=f"~{seed}")
    one_signal = _add_gate(gates, "NAND", [seed, not_seed], comment="Constante 1")
    if value:
        return _build_network_payload("NAND", gates, one_signal)

    zero_signal = _add_gate(gates, "NAND", [one_signal, one_signal], comment="Constante 0")
    return _build_network_payload("NAND", gates, zero_signal)


def _make_constant_nor_network(value: bool, variable_names: list[str]) -> dict[str, Any]:
    if not variable_names:
        return _build_network_payload(
            "NOR",
            [],
            "1" if value else "0",
            notes=[
                "Función constante sin variables: en hardware real se requiere una fuente fija de 1/0.",
            ],
        )

    seed = variable_names[0]
    gates: list[dict[str, Any]] = []
    not_seed = _add_gate(gates, "NOR", [seed, seed], comment=f"~{seed}")
    zero_signal = _add_gate(gates, "NOR", [seed, not_seed], comment="Constante 0")
    if not value:
        return _build_network_payload("NOR", gates, zero_signal)

    one_signal = _add_gate(gates, "NOR", [zero_signal, zero_signal], comment="Constante 1")
    return _build_network_payload("NOR", gates, one_signal)


def _synthesize_nand_network(expression: Any, variable_names: list[str]) -> dict[str, Any]:
    dnf, terms = _extract_dnf_terms(expression)
    if dnf == true:
        return _make_constant_nand_network(True, variable_names)
    if dnf == false:
        return _make_constant_nand_network(False, variable_names)

    gates: list[dict[str, Any]] = []
    inverted_cache: dict[str, str] = {}

    def invert_variable(name: str) -> str:
        if name not in inverted_cache:
            inverted_cache[name] = _add_gate(
                gates, "NAND", [name, name], comment=f"~{name}"
            )
        return inverted_cache[name]

    def literal_signal(literal: tuple[str, bool]) -> str:
        name, negated = literal
        if negated:
            return invert_variable(name)
        return name

    # F = P1 + P2 + ... + Pn, where each Pi is a product term.
    if len(terms) == 1:
        term = terms[0]
        if len(term) == 1:
            return _build_network_payload("NAND", gates, literal_signal(term[0]))

        literal_signals = [literal_signal(literal) for literal in term]
        term_label = " & ".join(_literal_text(literal) for literal in term)
        neg_term = _add_gate(gates, "NAND", literal_signals, comment=f"~({term_label})")
        output = _add_gate(gates, "NAND", [neg_term, neg_term], comment=term_label)
        return _build_network_payload("NAND", gates, output)

    complemented_terms: list[str] = []
    for term in terms:
        if len(term) == 1:
            name, negated = term[0]
            complemented_terms.append(name if negated else invert_variable(name))
            continue

        literal_signals = [literal_signal(literal) for literal in term]
        term_label = " & ".join(_literal_text(literal) for literal in term)
        complemented_terms.append(
            _add_gate(gates, "NAND", literal_signals, comment=f"~({term_label})")
        )

    output = _add_gate(gates, "NAND", complemented_terms, comment="OR de términos")
    return _build_network_payload("NAND", gates, output)


def _synthesize_nor_network(expression: Any, variable_names: list[str]) -> dict[str, Any]:
    cnf, clauses = _extract_cnf_clauses(expression)
    if cnf == true:
        return _make_constant_nor_network(True, variable_names)
    if cnf == false:
        return _make_constant_nor_network(False, variable_names)

    gates: list[dict[str, Any]] = []
    inverted_cache: dict[str, str] = {}

    def invert_variable(name: str) -> str:
        if name not in inverted_cache:
            inverted_cache[name] = _add_gate(
                gates, "NOR", [name, name], comment=f"~{name}"
            )
        return inverted_cache[name]

    def literal_signal(literal: tuple[str, bool]) -> str:
        name, negated = literal
        if negated:
            return invert_variable(name)
        return name

    # F = C1 * C2 * ... * Cn, where each Cj is a sum clause.
    if len(clauses) == 1:
        clause = clauses[0]
        if len(clause) == 1:
            return _build_network_payload("NOR", gates, literal_signal(clause[0]))

        literal_signals = [literal_signal(literal) for literal in clause]
        clause_label = " | ".join(_literal_text(literal) for literal in clause)
        neg_clause = _add_gate(
            gates, "NOR", literal_signals, comment=f"~({clause_label})"
        )
        output = _add_gate(gates, "NOR", [neg_clause, neg_clause], comment=clause_label)
        return _build_network_payload("NOR", gates, output)

    complemented_clauses: list[str] = []
    for clause in clauses:
        if len(clause) == 1:
            name, negated = clause[0]
            complemented_clauses.append(name if negated else invert_variable(name))
            continue

        literal_signals = [literal_signal(literal) for literal in clause]
        clause_label = " | ".join(_literal_text(literal) for literal in clause)
        complemented_clauses.append(
            _add_gate(gates, "NOR", literal_signals, comment=f"~({clause_label})")
        )

    output = _add_gate(gates, "NOR", complemented_clauses, comment="AND de cláusulas")
    return _build_network_payload("NOR", gates, output)


def synthesize_gates(expression: Any, variable_names: list[str]) -> dict[str, Any]:
    def walk(node: Any) -> dict[str, int]:
        counters = {
            "and": 0,
            "or": 0,
            "not": 0,
        }

        if node in {true, false} or isinstance(node, Symbol):
            return counters

        func = getattr(node, "func", None)
        args = list(getattr(node, "args", []))
        if not args:
            return counters

        for arg in args:
            _accumulate_gate_counters(counters, walk(arg))

        if func is Not:
            counters["not"] += 1
            return counters

        if func is And:
            counters["and"] += 1
            return counters

        if func is Or:
            counters["or"] += 1
            return counters

        counters["or"] += 1
        return counters

    counts = walk(expression)
    total_aon = counts["and"] + counts["or"] + counts["not"]
    nand_network = _synthesize_nand_network(expression, variable_names)
    nor_network = _synthesize_nor_network(expression, variable_names)

    return {
        "and_or_not": {
            "and": counts["and"],
            "or": counts["or"],
            "not": counts["not"],
            "total": total_aon,
        },
        "nand_only": nand_network,
        "nor_only": nor_network,
    }


def build_expression_circuit(expression: Any, variable_names: list[str]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    node_by_key: dict[Any, str] = {}
    level_by_id: dict[str, int] = {}
    edge_set: set[tuple[str, str]] = set()
    sequence = 0

    def create_node(key: Any, label: str, node_type: str, level: int) -> str:
        nonlocal sequence
        sequence += 1
        node_id = f"n{sequence}"
        node_by_key[key] = node_id
        level_by_id[node_id] = level
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "type": node_type,
                "level": level,
            }
        )
        return node_id

    def walk(node: Any) -> str:
        if node in node_by_key:
            return node_by_key[node]

        if node == true:
            return create_node(node, "1", "const", 0)
        if node == false:
            return create_node(node, "0", "const", 0)
        if isinstance(node, Symbol):
            return create_node(node, str(node), "input", 0)

        args = list(getattr(node, "args", []))
        child_ids = [walk(arg) for arg in args]
        child_levels = [level_by_id[child_id] for child_id in child_ids]
        level = (max(child_levels) + 1) if child_levels else 1

        func = getattr(node, "func", None)
        if func is Not:
            label = "NOT"
            node_type = "not"
        elif func is And:
            label = "AND"
            node_type = "and"
        elif func is Or:
            label = "OR"
            node_type = "or"
        elif func is Xor:
            label = "XOR"
            node_type = "xor"
        else:
            label = str(getattr(func, "__name__", "GATE")).upper()
            node_type = "gate"

        node_id = create_node(node, label, node_type, level)
        for child_id in child_ids:
            edge_key = (child_id, node_id)
            if edge_key not in edge_set:
                edges.append({"from": child_id, "to": node_id})
                edge_set.add(edge_key)

        return node_id

    output_source = walk(expression)
    output_level = level_by_id[output_source] + 1
    output_id = create_node(("output", output_source), "F", "output", output_level)
    edges.append({"from": output_source, "to": output_id})

    return {
        "title": "Circuito de la expresión simplificada (AND/OR/NOT)",
        "nodes": nodes,
        "edges": edges,
        "output_node": output_id,
        "output_source": output_source,
        "variables": variable_names,
    }


def build_gate_network_circuit(
    network: dict[str, Any], variable_names: list[str], title: str
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []
    node_by_signal: dict[str, str] = {}
    level_by_signal: dict[str, int] = {}
    edge_set: set[tuple[str, str]] = set()
    sequence = 0

    def create_signal_node(signal: str, label: str, node_type: str, level: int) -> str:
        nonlocal sequence
        sequence += 1
        node_id = f"n{sequence}"
        node_by_signal[signal] = node_id
        level_by_signal[signal] = level
        nodes.append(
            {
                "id": node_id,
                "label": label,
                "type": node_type,
                "level": level,
            }
        )
        return node_id

    def ensure_signal(signal: str) -> str:
        if signal in node_by_signal:
            return node_by_signal[signal]

        if signal in {"0", "1"}:
            return create_signal_node(signal, signal, "const", 0)
        if signal in variable_names:
            return create_signal_node(signal, signal, "input", 0)

        return create_signal_node(signal, signal, "wire", 0)

    for variable in variable_names:
        ensure_signal(variable)

    for gate in network.get("gates", []):
        gate_name = str(gate.get("name", ""))
        gate_type = str(gate.get("type", "GATE")).upper()
        gate_inputs = [str(item) for item in gate.get("inputs", [])]
        input_ids = [ensure_signal(signal) for signal in gate_inputs]
        input_levels = [level_by_signal.get(signal, 0) for signal in gate_inputs]
        gate_level = (max(input_levels) + 1) if input_levels else 1

        gate_node_id = create_signal_node(gate_name, gate_type, "gate", gate_level)
        for input_id in input_ids:
            edge_key = (input_id, gate_node_id)
            if edge_key not in edge_set:
                edges.append({"from": input_id, "to": gate_node_id})
                edge_set.add(edge_key)

    output_source = str(network.get("output_signal", "0"))
    output_source_id = ensure_signal(output_source)
    output_level = level_by_signal.get(output_source, 0) + 1
    output_id = create_signal_node("__out__", "F", "output", output_level)
    edges.append({"from": output_source_id, "to": output_id})

    return {
        "title": title,
        "nodes": nodes,
        "edges": edges,
        "output_node": output_id,
        "output_source": output_source,
        "variables": variable_names,
    }


def _canonical_from_terms(
    variable_names: list[str], minterms: list[int], dontcares: list[int]
) -> Any:
    if not variable_names:
        return true if minterms else false
    bool_symbols = make_boolean_symbols(variable_names)
    return SOPform(bool_symbols, minterms, dontcares)


def _ordered_union(first: list[str], second: list[str]) -> list[str]:
    merged: list[str] = []
    for item in [*first, *second]:
        if item not in merged:
            merged.append(item)
    return merged


def _validate_simplify_payload(payload: dict[str, Any]) -> tuple[str, str, list[str], list[int]]:
    mode = str(payload.get("mode", "equation")).strip().lower()
    result_form = str(payload.get("result_form", "sop")).strip().lower()

    if mode not in INPUT_MODES:
        raise ValueError("Modo inválido. Use: equation, table, minterms o maxterms.")
    if result_form not in RESULT_FORMS:
        raise ValueError("Formato inválido. Use 'sop' o 'pos'.")

    variables = parse_variables(payload.get("variables", ""))
    dontcares = parse_index_list(payload.get("dontcares", ""))
    return mode, result_form, variables, dontcares


@app.get("/")
def home():
    return render_template_string(INDEX_HTML)


@app.post("/api/simplify")
def simplify_endpoint():
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict(flat=True)

    try:
        mode, result_form, variables, dontcares = _validate_simplify_payload(payload)

        expression: Any
        simplified: Any
        terms: dict[str, list[int]]
        truth_table: dict[str, Any]

        if mode == "equation":
            expression, inferred_variables = parse_equation(
                str(payload.get("equation", "")), variables
            )
            if not variables:
                variables = inferred_variables

            form = "dnf" if result_form == "sop" else "cnf"
            simplified = simplify_boolean_form(expression, form)
            terms = derive_terms_from_expression(expression, variables)
            truth_table = build_truth_table_from_expression(expression, variables)
            equivalence = compare_expressions(
                expression,
                simplified,
                variables,
                method="auto",
                sample_size=DEFAULT_SAMPLE_SIZE,
            )
        elif mode == "table":
            parsed_table = parse_table(str(payload.get("table", "")))
            if variables and len(variables) != len(parsed_table.variables):
                raise ValueError(
                    "La cantidad de variables manuales no coincide con la tabla."
                )
            if not variables:
                variables = parsed_table.variables

            merged_dontcares = sorted(set(parsed_table.dontcares).union(dontcares))
            terms = summarize_terms(variables, parsed_table.minterms, merged_dontcares)
            simplified = simplify_from_minterms(
                variables, terms["minterms"], terms["dontcares"], result_form
            )
            expression = _canonical_from_terms(
                variables, terms["minterms"], terms["dontcares"]
            )
            truth_table = build_truth_table_from_terms(
                variables, terms["minterms"], terms["dontcares"]
            )
            equivalence = verify_expression_against_terms(
                simplified,
                variables,
                terms["minterms"],
                terms["dontcares"],
            )
        elif mode == "minterms":
            if not variables:
                raise ValueError("Debe indicar las variables para usar minitérminos.")

            minterms = parse_index_list(payload.get("terms", ""))
            terms = summarize_terms(variables, minterms, dontcares)
            simplified = simplify_from_minterms(
                variables, terms["minterms"], terms["dontcares"], result_form
            )
            expression = _canonical_from_terms(
                variables, terms["minterms"], terms["dontcares"]
            )
            truth_table = build_truth_table_from_terms(
                variables, terms["minterms"], terms["dontcares"]
            )
            equivalence = verify_expression_against_terms(
                simplified,
                variables,
                terms["minterms"],
                terms["dontcares"],
            )
        else:
            if not variables:
                raise ValueError("Debe indicar las variables para usar maxitérminos.")

            maxterms = parse_index_list(payload.get("terms", ""))
            minterms = minterms_from_maxterms(variables, maxterms, dontcares)
            terms = summarize_terms(variables, minterms, dontcares)
            simplified = simplify_from_minterms(
                variables, terms["minterms"], terms["dontcares"], result_form
            )
            expression = _canonical_from_terms(
                variables, terms["minterms"], terms["dontcares"]
            )
            truth_table = build_truth_table_from_terms(
                variables, terms["minterms"], terms["dontcares"]
            )
            equivalence = verify_expression_against_terms(
                simplified,
                variables,
                terms["minterms"],
                terms["dontcares"],
            )

        latex_expression = latex(simplified)
        gate_synthesis = synthesize_gates(simplified, variables)
        circuits = {
            "aon": build_expression_circuit(simplified, variables),
            "nand": build_gate_network_circuit(
                gate_synthesis["nand_only"], variables, "Circuito con solo NAND"
            ),
            "nor": build_gate_network_circuit(
                gate_synthesis["nor_only"], variables, "Circuito con solo NOR"
            ),
        }

        return jsonify(
            {
                "ok": True,
                "mode": mode,
                "result_form": result_form,
                "variables": variables,
                "expression": str(expression),
                "simplified_expression": str(simplified),
                "latex_expression": latex_expression,
                "terms": terms,
                "truth_table": truth_table,
                "equivalence": equivalence,
                "gate_synthesis": gate_synthesis,
                "circuits": circuits,
            }
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive fallback
        return jsonify(
            {
                "ok": False,
                "error": f"Error inesperado al simplificar la función: {exc}",
            }
        ), 500


@app.post("/api/equivalence")
def equivalence_endpoint():
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict(flat=True)

    try:
        raw_expression_a = str(payload.get("expression_a", "")).strip()
        raw_expression_b = str(payload.get("expression_b", "")).strip()
        if not raw_expression_a or not raw_expression_b:
            raise ValueError("Debe ingresar ambas expresiones para verificar equivalencia.")

        method = str(payload.get("method", "auto")).strip().lower()
        if method not in EQUIVALENCE_METHODS:
            raise ValueError("Método inválido. Use auto, exhaustive o sample.")

        sample_size_raw = str(payload.get("sample_size", DEFAULT_SAMPLE_SIZE)).strip()
        try:
            sample_size = int(sample_size_raw)
        except ValueError as exc:
            raise ValueError("sample_size debe ser un entero válido.") from exc

        explicit_variables = parse_variables(payload.get("variables", ""))

        if explicit_variables:
            expression_a, _ = parse_equation(raw_expression_a, explicit_variables)
            expression_b, _ = parse_equation(raw_expression_b, explicit_variables)
            variables = explicit_variables
        else:
            expression_a, inferred_a = parse_equation(raw_expression_a, [])
            expression_b, inferred_b = parse_equation(raw_expression_b, [])
            variables = _ordered_union(inferred_a, inferred_b)
            expression_a, _ = parse_equation(raw_expression_a, variables)
            expression_b, _ = parse_equation(raw_expression_b, variables)

        result = compare_expressions(
            expression_a,
            expression_b,
            variables,
            method=method,
            sample_size=sample_size,
        )

        return jsonify(
            {
                "ok": True,
                "variables": variables,
                "expression_a": str(expression_a),
                "expression_b": str(expression_b),
                "latex_a": latex(expression_a),
                "latex_b": latex(expression_b),
                "result": result,
            }
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive fallback
        return jsonify(
            {
                "ok": False,
                "error": f"Error inesperado al verificar equivalencia: {exc}",
            }
        ), 500


@app.post("/api/convert")
def convert_endpoint():
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict(flat=True)

    try:
        direction = str(payload.get("direction", "equation_to_table")).strip().lower()
        if direction not in CONVERT_DIRECTIONS:
            raise ValueError(
                "Dirección inválida. Use equation_to_table o table_to_equation."
            )

        variables = parse_variables(payload.get("variables", ""))
        dontcares = parse_index_list(payload.get("dontcares", ""))

        if direction == "equation_to_table":
            expression, inferred = parse_equation(str(payload.get("equation", "")), variables)
            if not variables:
                variables = inferred

            terms = derive_terms_from_expression(expression, variables)
            truth_table = build_truth_table_from_expression(expression, variables)
            sop_expr = simplify_boolean_form(expression, "dnf")
            pos_expr = simplify_boolean_form(expression, "cnf")

            return jsonify(
                {
                    "ok": True,
                    "direction": direction,
                    "variables": variables,
                    "truth_table": truth_table,
                    "terms": terms,
                    "sop_expression": str(sop_expr),
                    "pos_expression": str(pos_expr),
                    "sop_latex": latex(sop_expr),
                    "pos_latex": latex(pos_expr),
                }
            )

        parsed_table = parse_table(str(payload.get("table", "")))
        if variables and len(variables) != len(parsed_table.variables):
            raise ValueError("La cantidad de variables no coincide con la tabla ingresada.")
        if not variables:
            variables = parsed_table.variables

        merged_dontcares = sorted(set(parsed_table.dontcares).union(dontcares))
        terms = summarize_terms(variables, parsed_table.minterms, merged_dontcares)
        sop_expr = simplify_from_minterms(variables, terms["minterms"], terms["dontcares"], "sop")
        pos_expr = simplify_from_minterms(variables, terms["minterms"], terms["dontcares"], "pos")
        truth_table = build_truth_table_from_terms(
            variables, terms["minterms"], terms["dontcares"]
        )

        return jsonify(
            {
                "ok": True,
                "direction": direction,
                "variables": variables,
                "truth_table": truth_table,
                "terms": terms,
                "sop_expression": str(sop_expr),
                "pos_expression": str(pos_expr),
                "sop_latex": latex(sop_expr),
                "pos_latex": latex(pos_expr),
            }
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive fallback
        return jsonify(
            {
                "ok": False,
                "error": f"Error inesperado en conversión: {exc}",
            }
        ), 500


INDEX_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Simplificador booleano</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
      --canvas: #eaf4fa;
      --paper: #fbfdff;
      --surface: #f3f8fc;
      --surface-blue: #e3f2fb;
      --ink: #0a2134;
      --muted: #587083;
      --faint: #7890a2;
      --line: #c4d7e4;
      --line-strong: #9fbccd;
      --blue: #0769a5;
      --blue-dark: #06466d;
      --blue-bright: #28a6d4;
      --blue-soft: #cfeaf8;
      --green: #18734b;
      --green-soft: #e8f7ef;
      --red: #a43845;
      --red-soft: #fff0f2;
      --shadow: 0 24px 70px rgba(33, 76, 104, 0.13);
    }

    * { box-sizing: border-box; }

    html { scroll-behavior: smooth; }

    body {
      margin: 0;
      font-family: "Space Grotesk", "Avenir Next", sans-serif;
      color: var(--ink);
      background:
        linear-gradient(rgba(31, 109, 153, 0.055) 1px, transparent 1px),
        linear-gradient(90deg, rgba(31, 109, 153, 0.055) 1px, transparent 1px),
        radial-gradient(circle at 12% 2%, rgba(89, 194, 228, 0.24), transparent 28rem),
        radial-gradient(circle at 88% 75%, rgba(75, 150, 205, 0.14), transparent 34rem),
        var(--canvas);
      background-size: 32px 32px, 32px 32px, auto, auto, auto;
      min-height: 100vh;
      padding: 2.75rem 1.25rem 3.25rem;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 1.25rem;
    }

    .container {
      width: min(1120px, 100%);
      background: var(--paper);
      border: 1px solid rgba(133, 169, 190, 0.8);
      border-radius: 26px;
      box-shadow: var(--shadow);
      overflow: hidden;
      animation: page-in 0.55s cubic-bezier(0.2, 0.75, 0.25, 1) both;
    }

    .header {
      position: relative;
      padding: 2.5rem 2.75rem 2.65rem;
      overflow: hidden;
      background:
        linear-gradient(110deg, rgba(255, 255, 255, 0.97), rgba(230, 245, 253, 0.92)),
        var(--paper);
      border-bottom: 1px solid var(--line);
    }

    .header::after {
      content: '';
      position: absolute;
      width: 280px;
      height: 280px;
      right: -92px;
      top: -126px;
      border: 46px solid rgba(41, 161, 207, 0.12);
      border-radius: 50%;
      pointer-events: none;
    }

    .header-layout {
      position: relative;
      z-index: 1;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: end;
      gap: 2rem;
    }

    .brand-line,
    .section-number,
    .result-kicker,
    .method-label {
      font-family: "IBM Plex Mono", monospace;
      text-transform: uppercase;
      letter-spacing: 0.13em;
      font-size: 0.72rem;
      font-weight: 600;
    }

    .brand-line {
      display: inline-flex;
      align-items: center;
      gap: 0.65rem;
      margin-bottom: 1rem;
      color: var(--blue);
    }

    .brand-line::before {
      content: '';
      width: 1.8rem;
      height: 2px;
      background: var(--blue-bright);
      box-shadow: 0 6px 0 rgba(40, 166, 212, 0.45);
    }

    .header h1 {
      max-width: 760px;
      margin: 0;
      font-size: clamp(2.3rem, 5vw, 4.65rem);
      line-height: 0.96;
      letter-spacing: -0.065em;
      font-weight: 700;
      text-wrap: balance;
    }

    .header p {
      max-width: 700px;
      margin: 1.15rem 0 0;
      color: var(--muted);
      line-height: 1.65;
      font-size: 1rem;
    }

    .method-card {
      min-width: 212px;
      padding: 1rem 1.1rem;
      border: 1px solid var(--line-strong);
      border-radius: 15px;
      background: rgba(251, 253, 255, 0.74);
      box-shadow: 8px 8px 0 rgba(82, 156, 195, 0.12);
    }

    .method-label {
      display: block;
      color: var(--faint);
      margin-bottom: 0.4rem;
    }

    .method-card strong {
      display: block;
      font-size: 0.96rem;
      color: var(--blue-dark);
    }

    .block {
      padding: 2.2rem 2.75rem 2.7rem;
      display: grid;
      gap: 1.35rem;
      align-content: start;
    }

    .section-heading {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 1rem;
      align-items: start;
    }

    .section-number {
      display: grid;
      place-items: center;
      width: 2.5rem;
      height: 2.5rem;
      border: 1px solid var(--line-strong);
      border-radius: 50%;
      color: var(--blue);
      background: var(--surface-blue);
    }

    .section-heading h2,
    .block > h2 {
      margin: 0;
      color: var(--ink);
      font-size: clamp(1.35rem, 2vw, 1.75rem);
      line-height: 1.2;
      letter-spacing: -0.035em;
    }

    .section-heading p {
      margin: 0.38rem 0 0;
      color: var(--muted);
      line-height: 1.55;
      font-size: 0.92rem;
    }

    .logic-form {
      display: grid;
      gap: 1.15rem;
      margin-top: 0.25rem;
      padding: 1.35rem;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: linear-gradient(145deg, #ffffff, #f5faff);
    }

    .tools-block {
      gap: 1.1rem;
    }

    .tools-block details.module-frame.ondemand {
      margin: 0;
    }

    .tools-block .module-frame + .module-frame {
      margin-top: 0.25rem;
    }

    .module-frame {
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 17px;
      background: rgba(255, 255, 255, 0.72);
      padding: 1.05rem 1.15rem;
      display: grid;
      gap: 1rem;
      transition: border-color 0.2s ease, background 0.2s ease;
    }

    details.ondemand {
      display: block;
      border: 1px solid var(--line);
      border-radius: 15px;
      background: var(--surface);
      padding: 0.9rem 1rem;
      margin: 0;
    }

    details.module-frame.ondemand {
      margin: 0;
    }

    details.module-frame[open] {
      border-color: var(--line-strong);
      background: #fff;
    }

    details.ondemand > summary {
      cursor: pointer;
      font-size: 1.03rem;
      font-weight: 600;
      color: var(--ink);
      user-select: none;
      outline: none;
      list-style: none;
      display: flex;
      align-items: center;
      min-height: 2rem;
      gap: 0.75rem;
    }

    details.ondemand > summary::after {
      content: '';
      margin-left: auto;
      width: 0.55rem;
      height: 0.55rem;
      border-right: 1.5px solid var(--blue);
      border-bottom: 1.5px solid var(--blue);
      transform: rotate(45deg) translateY(-2px);
      transition: transform 0.2s ease;
    }

    details.ondemand > summary::-webkit-details-marker {
      display: none;
    }

    details.ondemand[open] > summary {
      margin-bottom: 1rem;
    }

    details.ondemand[open] > summary::after {
      transform: rotate(225deg) translate(-2px, -2px);
    }

    details.ondemand.plain {
      border: 0;
      background: transparent;
      padding: 0;
    }

    details.ondemand.plain > summary {
      padding: 0.1rem 0.15rem;
    }

    .module-inner {
      border: 0;
      border-radius: 0;
      background: transparent;
      padding: 0;
      display: grid;
      gap: 1rem;
      margin: 0 0.2rem;
    }

    .module-frame > .result {
      margin: 0.35rem 0.2rem 0;
    }

    .synthesis-title {
      margin: 0 0 1rem;
    }

    .panel h2 {
      color: var(--ink);
      font-size: 1.18rem;
      line-height: 1.25;
      letter-spacing: -0.025em;
    }

    .panel hr {
      width: 100%;
      margin: 0.5rem 0;
      border: 0;
      border-top: 1px solid var(--line);
    }

    .row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1rem;
    }

    .row-3 {
      display: grid;
      grid-template-columns: 2fr 1fr 1fr;
      gap: 1rem;
    }

    label {
      display: grid;
      gap: 0.5rem;
      font-size: 0.82rem;
      color: var(--muted);
      font-weight: 600;
      letter-spacing: 0.015em;
      align-content: start;
    }

    input, select, textarea, button {
      font: inherit;
    }

    input, select, textarea {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 11px;
      padding: 0.78rem 0.85rem;
      background: rgba(255, 255, 255, 0.95);
      color: var(--ink);
      box-shadow: 0 1px 0 rgba(255, 255, 255, 0.8);
      transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
    }

    input::placeholder,
    textarea::placeholder {
      color: #8da2b1;
      opacity: 1;
      font-weight: 400;
    }

    input:focus, select:focus, textarea:focus {
      outline: none;
      border-color: var(--blue);
      background: #fff;
      box-shadow: 0 0 0 4px rgba(37, 153, 202, 0.14);
    }

    textarea {
      min-height: 132px;
      resize: vertical;
      line-height: 1.5;
      font-family: "IBM Plex Mono", monospace;
      font-size: 0.88rem;
    }

    .hidden { display: none !important; }

    .hint {
      font-size: 0.78rem;
      line-height: 1.5;
      color: var(--faint);
      margin-top: 0.1rem;
      font-weight: 400;
    }

    .placeholder-note {
      display: inline-flex;
      align-items: center;
      width: fit-content;
      margin: 0;
      padding: 0.45rem 0.65rem;
      border-left: 2px solid var(--blue-bright);
      background: rgba(226, 242, 251, 0.62);
    }

    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 0.7rem;
      align-items: center;
    }

    button {
      appearance: none;
      border: 1px solid var(--blue-dark);
      border-radius: 10px;
      padding: 0.76rem 1.05rem;
      cursor: pointer;
      background: var(--blue-dark);
      color: #fff;
      font-weight: 600;
      width: fit-content;
      box-shadow: 4px 4px 0 rgba(40, 166, 212, 0.3);
      transition: transform 0.16s ease, box-shadow 0.16s ease, background 0.16s ease;
    }

    button[type="submit"]::after {
      content: '→';
      display: inline-block;
      margin-left: 0.65rem;
      transition: transform 0.16s ease;
    }

    button:hover {
      background: var(--blue);
      transform: translate(-1px, -1px);
      box-shadow: 6px 6px 0 rgba(40, 166, 212, 0.3);
    }

    button[type="submit"]:hover::after {
      transform: translateX(3px);
    }

    button:active {
      transform: translate(2px, 2px);
      box-shadow: 2px 2px 0 rgba(40, 166, 212, 0.28);
    }

    button:focus-visible,
    summary:focus-visible,
    a:focus-visible {
      outline: 3px solid rgba(40, 166, 212, 0.35);
      outline-offset: 3px;
    }

    button.secondary {
      background: #fff;
      color: var(--blue-dark);
      border: 1px solid var(--line-strong);
      font-weight: 600;
      box-shadow: none;
    }

    button.secondary:hover {
      background: var(--surface-blue);
      box-shadow: none;
    }

    .result {
      border-radius: 16px;
      border: 1px solid var(--line);
      border-left: 4px solid var(--blue);
      padding: 1.2rem;
      background: #f7fbfe;
      display: grid;
      gap: 0.8rem;
      word-break: break-word;
      align-content: start;
    }

    #equivalenceResult {
      margin-top: 1rem;
    }

    .ok {
      background: linear-gradient(145deg, #f9fdff, #edf8fd);
      border-color: #9fc9de;
    }

    .error {
      border-color: #e4a9b0;
      border-left-color: var(--red);
      background: var(--red-soft);
      color: var(--red);
    }

    .result-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(220px, auto);
      gap: 1.25rem;
      align-items: end;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--line);
    }

    .result-kicker {
      display: block;
      color: var(--blue);
      margin-bottom: 0.35rem;
    }

    .result-format {
      color: var(--muted);
      font-size: 0.88rem;
    }

    code.result-expression {
      display: flex;
      align-items: baseline;
      gap: 0.7rem;
      min-width: min(100%, 240px);
      padding: 0.25rem 0 0.35rem 1rem;
      background: transparent;
      color: var(--blue-dark);
      border: 0;
      border-left: 3px solid var(--blue-bright);
      border-radius: 0;
      font-family: "IBM Plex Mono", "Consolas", monospace;
      font-size: clamp(1.2rem, 2.4vw, 1.65rem);
      font-weight: 500;
      line-height: 1.35;
      text-align: left;
      box-shadow: none;
      overflow-wrap: anywhere;
    }

    code.result-expression::before {
      content: 'F =';
      flex: 0 0 auto;
      color: var(--faint);
      font-size: inherit;
      font-family: inherit;
      font-weight: inherit;
      line-height: inherit;
      letter-spacing: 0;
    }

    .result-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 1px;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 11px;
      background: var(--line);
    }

    .result-item {
      min-width: 0;
      padding: 0.75rem 0.85rem;
      background: rgba(255, 255, 255, 0.8);
    }

    .result-item span {
      display: block;
      margin-bottom: 0.28rem;
      color: var(--faint);
      font-family: "IBM Plex Mono", monospace;
      font-size: 0.68rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .result-item strong {
      display: block;
      color: var(--ink);
      font-size: 0.88rem;
      font-weight: 600;
      overflow-wrap: anywhere;
    }

    .result-proof {
      display: flex;
      align-items: center;
      gap: 0.55rem;
      color: var(--green);
      font-size: 0.86rem;
      font-weight: 600;
    }

    .result-proof::before {
      content: '✓';
      display: grid;
      place-items: center;
      width: 1.35rem;
      height: 1.35rem;
      flex: 0 0 auto;
      border-radius: 50%;
      background: var(--green-soft);
    }

    .result-proof.failed {
      color: var(--red);
    }

    .result-proof.failed::before {
      content: '!';
      background: var(--red-soft);
    }

    .result-counterexample {
      padding: 0.75rem 0.85rem;
      border-radius: 9px;
      background: var(--red-soft);
      color: var(--red);
      font-size: 0.84rem;
    }

    .panel {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 1.15rem 1.2rem;
      background: rgba(247, 251, 254, 0.8);
      display: grid;
      gap: 0.85rem;
      align-content: start;
    }

    .mini-table {
      overflow: auto;
      max-height: 320px;
      border: 1px solid var(--line);
      border-radius: 11px;
      background: #fff;
    }

    .truth-table-gap {
      margin-top: 0.55rem;
    }

    table {
      border-collapse: collapse;
      width: 100%;
      font-size: 0.88rem;
    }

    th, td {
      border: 1px solid #d8e5ed;
      padding: 0.52rem 0.62rem;
      text-align: center;
      white-space: nowrap;
    }

    th {
      background: var(--surface-blue);
      color: var(--blue-dark);
      font-weight: 700;
      position: sticky;
      top: 0;
    }

    code {
      background: #e4f2fa;
      border: 1px solid #d0e5f1;
      border-radius: 5px;
      padding: 0.13rem 0.32rem;
      font-family: "IBM Plex Mono", "Consolas", monospace;
      font-size: 0.86em;
    }

    pre {
      margin: 0;
      background: #eef6fb;
      border-radius: 10px;
      padding: 0.85rem;
      overflow: auto;
      max-height: 260px;
      border: 1px solid var(--line);
      font-family: "IBM Plex Mono", "Consolas", monospace;
      font-size: 0.83rem;
      line-height: 1.55;
    }

    .circuit-toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem;
      align-items: center;
    }

    .circuit-radio-group {
      display: flex;
      flex-wrap: wrap;
      gap: 0.55rem;
      align-items: center;
      margin: 0;
      padding: 0;
      border: 0;
    }

    .circuit-radio {
      display: inline-flex;
      align-items: center;
      gap: 0.38rem;
      padding: 0.2rem 0.1rem;
      border: 0;
      border-radius: 0;
      background: transparent;
      color: var(--blue-dark);
      font-size: 0.9rem;
      cursor: pointer;
      user-select: none;
      white-space: nowrap;
    }

    .circuit-radio input[type="radio"] {
      margin: 0;
      width: 1rem;
      height: 1rem;
      accent-color: var(--blue);
    }

    .circuit-radio.active {
      background: transparent;
      color: var(--ink);
      font-weight: 700;
    }

    .circuit-wrap {
      overflow: auto;
      padding: 0;
      border: 0;
      border-radius: 0;
      background: transparent;
    }

    .circuit-svg text {
      font-family: "Space Grotesk", sans-serif;
      font-size: 12px;
      fill: #0f2527;
      user-select: none;
    }

    .circuit-wire {
      stroke: #496f87;
      stroke-width: 1.4;
      fill: none;
    }

    .circuit-shape {
      stroke: #28566f;
      stroke-width: 1.5;
      fill: #fafeff;
    }

    .circuit-io {
      stroke: #28566f;
      stroke-width: 1.3;
      fill: #f3f9ff;
    }

    .circuit-input {
      fill: #dff3fb;
    }

    .circuit-output {
      fill: #bde7f7;
    }

    .circuit-const {
      fill: #eef3ff;
    }

    .circuit-label {
      font-size: 11px;
      fill: #244a70;
    }

    .circuit-gate-label {
      font-size: 10px;
      fill: #244a70;
    }

    .footer {
      width: min(1120px, 100%);
      margin-top: 1.5rem;
      padding: 0.4rem 1.5rem;
      font-size: 0.82rem;
      color: var(--muted);
      text-align: center;
    }

    .footer a {
      color: var(--blue-dark);
      text-decoration: none;
      font-weight: 400;
    }

    .footer a:hover { color: var(--blue); }

    .tools-container {
      animation-delay: 0.08s;
    }

    @keyframes page-in {
      from { opacity: 0; transform: translateY(16px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 860px) {
      body { padding: 1.25rem 0.8rem 2.5rem; }
      .container { border-radius: 20px; }
      .header { padding: 2rem 1.35rem; }
      .header-layout { grid-template-columns: 1fr; align-items: start; gap: 1.5rem; }
      .method-card { min-width: 0; width: min(100%, 320px); }
      .block { padding: 1.5rem 1.35rem 1.8rem; }
      .row, .row-3 { grid-template-columns: 1fr; }
      .result-head { grid-template-columns: 1fr; align-items: start; }
      code.result-expression { width: 100%; text-align: left; }
      .result-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }

    @media (max-width: 540px) {
      .header h1 { font-size: 2.45rem; }
      .logic-form { padding: 1rem; border-radius: 14px; }
      .section-heading { gap: 0.75rem; }
      .section-number { width: 2.1rem; height: 2.1rem; }
      .result-grid { grid-template-columns: 1fr; }
      .result, .panel { padding: 1rem; }
      button { width: 100%; justify-content: center; }
      .actions { display: grid; grid-template-columns: 1fr; }
    }

    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after {
        scroll-behavior: auto !important;
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
      }
    }
  </style>
</head>
<body>
  <main class="container">
    <header class="header">
      <div class="header-layout">
        <div>
          <div class="brand-line">Laboratorio de lógica digital</div>
          <h1>Simplificador booleano</h1>
          <p>Reduce expresiones, tablas y términos canónicos; después explora su tabla de verdad y la síntesis del circuito.</p>
        </div>
        <aside class="method-card" aria-label="Método de simplificación">
          <span class="method-label">Método exacto</span>
          <strong>Quine‑McCluskey</strong>
        </aside>
      </div>
    </header>

    <section class="block">
      <div class="section-heading">
        <span class="section-number">01</span>
        <div>
          <h2>Simplificación lógica</h2>
          <p>Elige una representación de entrada y la forma normal que quieres obtener.</p>
        </div>
      </div>
      <div class="hint placeholder-note">Los textos en gris dentro de los campos son ejemplos; el campo sigue vacío hasta que escribas.</div>
      <form id="logicForm" class="logic-form">
        <div class="row">
          <label>
            Modo de entrada
            <select id="mode" name="mode">
              <option value="equation">Ecuación</option>
              <option value="table">Tabla de verdad</option>
              <option value="minterms">Lista de minitérminos</option>
              <option value="maxterms">Lista de maxitérminos</option>
            </select>
          </label>

          <label>
            Formato de salida
            <select id="result_form" name="result_form">
              <option value="sop">Suma de productos (SOP)</option>
              <option value="pos">Producto de sumas (POS)</option>
            </select>
          </label>
        </div>

        <label id="variablesField">
          Variables (separadas por coma)
          <input id="variables" name="variables" placeholder="A, B, C" />
          <span class="hint">Obligatorio para minitérminos y maxitérminos. Opcional para ecuación y tabla.</span>
        </label>

        <label id="equationField">
          Ecuación lógica
          <textarea id="equation" name="equation" placeholder="F = (A & ~B) | (B & C)"></textarea>
          <span class="hint">Operadores: <code>~</code>, <code>&</code>, <code>|</code>, <code>^</code>. También: <code>!</code>, <code>*</code>, <code>+</code>, <code>AND</code>, <code>OR</code>, <code>NOT</code>.</span>
        </label>

        <label id="tableField" class="hidden">
          Tabla (CSV o columnas por espacio)
          <textarea id="table" name="table" placeholder="A,B,C,F\n0,0,0,0\n0,0,1,1\n0,1,0,0\n0,1,1,1\n1,0,0,1\n1,0,1,1\n1,1,0,0\n1,1,1,1"></textarea>
          <span class="hint">La última columna es la salida (0, 1 o x para don't care).</span>
        </label>

        <label id="termsField" class="hidden">
          Lista de términos (índices)
          <input id="terms" name="terms" placeholder="1, 3, 5, 7" />
          <span class="hint">En minitérminos: índices donde F=1. En maxitérminos: índices donde F=0.</span>
        </label>

        <label>
          Don't care (opcional)
          <input id="dontcares" name="dontcares" placeholder="2, 6" />
        </label>

        <div class="actions">
          <button type="submit">Simplificar</button>
        </div>
      </form>

      <section id="result" class="result hidden" aria-live="polite"></section>
      <section id="truthPanel" class="panel hidden" aria-live="polite"></section>
      <section id="gatePanel" class="panel hidden" aria-live="polite"></section>
      <section id="circuitPanel" class="panel hidden" aria-live="polite"></section>
    </section>
  </main>

  <section class="container tools-container">
    <section class="block tools-block">
      <div class="section-heading">
        <span class="section-number">02</span>
        <div>
          <h2>Herramientas adicionales</h2>
          <p>Abre únicamente el módulo que necesites para comparar o transformar representaciones.</p>
        </div>
      </div>
    <details class="module-frame ondemand">
      <summary>Verificador de equivalencia</summary>
      <form id="equivalenceForm" class="module-inner">
        <div class="row">
          <label>
            Expresión 1
            <textarea id="expressionA" placeholder="(A & B) | (A & ~B)"></textarea>
          </label>

          <label>
            Expresión 2
            <textarea id="expressionB" placeholder="A"></textarea>
          </label>
        </div>

        <div class="row-3">
          <label>
            Variables (opcional)
            <input id="equivVariables" placeholder="A, B" />
            <span class="hint">Si se omite, se infieren de ambas expresiones.</span>
          </label>

          <label>
            Método
            <select id="equivMethod">
              <option value="auto">Auto</option>
              <option value="exhaustive">Exhaustivo</option>
              <option value="sample">Muestreo</option>
            </select>
          </label>

          <label>
            Tamaño de muestra
            <input id="equivSample" type="number" min="1" value="256" />
          </label>
        </div>

        <div class="actions">
          <button type="submit">Verificar equivalencia</button>
        </div>
      </form>

      <section id="equivalenceResult" class="result hidden" aria-live="polite"></section>
    </details>

    <details class="module-frame ondemand">
      <summary>Conversión automática</summary>
      <form id="convertForm" class="module-inner">
        <div class="row">
          <label>
            Dirección
            <select id="convertDirection">
              <option value="equation_to_table">Ecuación → Tabla</option>
              <option value="table_to_equation">Tabla → Ecuación</option>
            </select>
          </label>

          <label>
            Variables (opcional)
            <input id="convertVariables" placeholder="A, B, C" />
          </label>
        </div>

        <label id="convertEquationField">
          Ecuación
          <textarea id="convertEquation" placeholder="F = (A & ~B) | C"></textarea>
        </label>

        <label id="convertTableField" class="hidden">
          Tabla
          <textarea id="convertTable" placeholder="A,B,F\n0,0,0\n0,1,1\n1,0,1\n1,1,1"></textarea>
        </label>

        <label id="convertDontcaresField" class="hidden">
          Don't cares adicionales (opcional)
          <input id="convertDontcares" placeholder="2, 6" />
        </label>

        <div class="actions">
          <button type="submit">Convertir</button>
        </div>
      </form>

      <section id="convertResult" class="result hidden" aria-live="polite"></section>
    </details>
    </section>
  </section>
  <footer class="footer">&copy; 2026, <a href="https://isantosruiz.github.io/home/" target="_blank" rel="noopener noreferrer">Ildeberto de los Santos Ruiz</a></footer>

  <script>
    const modeElement = document.getElementById('mode');
    const equationField = document.getElementById('equationField');
    const tableField = document.getElementById('tableField');
    const termsField = document.getElementById('termsField');

    const resultBox = document.getElementById('result');
    const gatePanel = document.getElementById('gatePanel');
    const circuitPanel = document.getElementById('circuitPanel');
    const truthPanel = document.getElementById('truthPanel');
    const equivalenceResultBox = document.getElementById('equivalenceResult');
    const convertResultBox = document.getElementById('convertResult');

    const convertDirectionElement = document.getElementById('convertDirection');
    const convertEquationField = document.getElementById('convertEquationField');
    const convertTableField = document.getElementById('convertTableField');
    const convertDontcaresField = document.getElementById('convertDontcaresField');

    const CIRCUIT_OPTIONS = {
      aon: 'AND/OR/NOT',
      nand: 'Solo NAND',
      nor: 'Solo NOR',
    };
    let latestCircuits = null;
    let activeCircuitKey = 'aon';

    function escapeHtml(value) {
      return String(value)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
    }

    function formatList(values) {
      if (!values || values.length === 0) {
        return '(vacío)';
      }
      return values.join(', ');
    }

    function refreshVisibleFields() {
      const mode = modeElement.value;
      equationField.classList.toggle('hidden', mode !== 'equation');
      tableField.classList.toggle('hidden', mode !== 'table');
      termsField.classList.toggle('hidden', mode !== 'minterms' && mode !== 'maxterms');
    }

    function refreshConvertFields() {
      const direction = convertDirectionElement.value;
      convertEquationField.classList.toggle('hidden', direction !== 'equation_to_table');
      convertTableField.classList.toggle('hidden', direction !== 'table_to_equation');
      convertDontcaresField.classList.toggle('hidden', direction !== 'table_to_equation');
    }

    modeElement.addEventListener('change', refreshVisibleFields);
    convertDirectionElement.addEventListener('change', refreshConvertFields);
    refreshVisibleFields();
    refreshConvertFields();

    function resolveCircuitGateType(node) {
      const type = String(node.type || '').toLowerCase();
      const label = String(node.label || '').toUpperCase();

      if (type === 'not' || label === 'NOT') return 'NOT';
      if (type === 'and' || label === 'AND') return 'AND';
      if (type === 'or' || label === 'OR') return 'OR';
      if (type === 'xor' || label === 'XOR') return 'XOR';
      if (label === 'NAND') return 'NAND';
      if (label === 'NOR') return 'NOR';
      if (label === 'XNOR') return 'XNOR';
      return 'GENERIC';
    }

    function getNodeBox(node) {
      const type = String(node.type || '').toLowerCase();
      if (type === 'input' || type === 'output' || type === 'const' || type === 'wire') {
        return { width: 26, height: 26, io: true };
      }

      const gateType = resolveCircuitGateType(node);
      if (gateType === 'NOT') return { width: 78, height: 54, io: false };
      if (gateType === 'AND') return { width: 68, height: 58, io: false };
      if (gateType === 'NAND') return { width: 84, height: 58, io: false };
      if (gateType === 'OR' || gateType === 'XOR' || gateType === 'NOR' || gateType === 'XNOR') {
        return { width: 96, height: 58, io: false };
      }
      return { width: 84, height: 50, io: false };
    }

    function drawGateSymbol(gateType, x, y, width, height, label) {
      const elements = [];
      const cy = y + (height / 2);
      const bubbleRadius = 4.8;
      const hasBubble = gateType === 'NAND' || gateType === 'NOR' || gateType === 'XNOR';
      let leftX = x;
      let outputBaseX = x + width;
      let inputBoundary = { kind: 'line', x: leftX };

      if (gateType === 'NOT') {
        const bodyWidth = width - 14;
        const tipX = x + bodyWidth;
        const points = `${x},${y} ${x},${y + height} ${tipX},${cy}`;
        const bubbleCx = tipX + bubbleRadius;
        elements.push(`<polygon class="circuit-shape" points="${points}"></polygon>`);
        elements.push(`<circle class="circuit-shape" cx="${bubbleCx}" cy="${cy}" r="${bubbleRadius}"></circle>`);
        leftX = x;
        outputBaseX = bubbleCx + bubbleRadius;
        inputBoundary = { kind: 'line', x: leftX };
      } else if (gateType === 'AND' || gateType === 'NAND') {
        const bodyWidth = width - (hasBubble ? (bubbleRadius * 2 + 5) : 0);
        const radius = height / 2;
        const arcLeftX = x + bodyWidth - radius;
        const path = [
          `M ${x} ${y}`,
          `L ${arcLeftX} ${y}`,
          `A ${radius} ${radius} 0 0 1 ${arcLeftX} ${y + height}`,
          `L ${x} ${y + height}`,
          'Z',
        ].join(' ');
        elements.push(`<path class="circuit-shape" d="${path}"></path>`);
        leftX = x;
        outputBaseX = x + bodyWidth;
        inputBoundary = { kind: 'line', x: leftX };
      } else if (gateType === 'OR' || gateType === 'NOR' || gateType === 'XOR' || gateType === 'XNOR') {
        const bodyWidth = width - (hasBubble ? (bubbleRadius * 2 + 5) : 0);
        const x1 = x + bodyWidth * 0.16;
        const x2 = x + bodyWidth * 0.60;
        const x3 = x + bodyWidth * 0.92;
        const backCurveX = x + bodyWidth * 0.31;
        const path = [
          `M ${x1} ${y}`,
          `Q ${x2} ${y} ${x3} ${cy}`,
          `Q ${x2} ${y + height} ${x1} ${y + height}`,
          `Q ${backCurveX} ${cy} ${x1} ${y}`,
          'Z',
        ].join(' ');
        elements.push(`<path class="circuit-shape" d="${path}"></path>`);
        if (gateType === 'XOR' || gateType === 'XNOR') {
          const xorOffset = 7;
          const xorPath = [
            `M ${x1 - xorOffset} ${y}`,
            `Q ${backCurveX - xorOffset} ${cy} ${x1 - xorOffset} ${y + height}`,
          ].join(' ');
          elements.push(`<path class="circuit-shape" d="${xorPath}" fill="none"></path>`);
        }
        leftX = x + bodyWidth * 0.16;
        outputBaseX = x + bodyWidth * 0.92;
        inputBoundary = {
          kind: 'quadratic',
          xStart: x1,
          xCtrl: backCurveX,
          top: y,
          height,
        };
      } else {
        elements.push(
          `<rect class="circuit-shape" x="${x}" y="${y}" width="${width}" height="${height}" rx="8" ry="8"></rect>`
        );
        elements.push(
          `<text class="circuit-gate-label" x="${x + width / 2}" y="${y + height / 2 + 4}" text-anchor="middle">${escapeHtml(label)}</text>`
        );
        leftX = x;
        outputBaseX = x + width;
        inputBoundary = { kind: 'line', x: leftX };
      }

      if (hasBubble) {
        const bubbleCx = outputBaseX + bubbleRadius;
        elements.push(`<circle class="circuit-shape" cx="${bubbleCx}" cy="${cy}" r="${bubbleRadius}"></circle>`);
        outputBaseX = bubbleCx + bubbleRadius;
      }

      let outputLead = 0;
      if (gateType === 'AND') {
        outputLead = 3;
      } else if (gateType === 'NAND') {
        outputLead = 5;
      } else if (gateType === 'OR' || gateType === 'NOR' || gateType === 'XOR' || gateType === 'XNOR' || gateType === 'NOT') {
        outputLead = 8;
      }
      if (outputLead > 0) {
        const outputX = outputBaseX + outputLead;
        elements.push(`<line class="circuit-shape" x1="${outputBaseX}" y1="${cy}" x2="${outputX}" y2="${cy}"></line>`);
        outputBaseX = outputX;
      }

      return {
        markup: elements.join(''),
        leftX,
        rightX: outputBaseX,
        inputBoundary,
      };
    }

    function buildCircuitSvg(circuit) {
      if (!circuit || !Array.isArray(circuit.nodes) || circuit.nodes.length === 0) {
        return '<div class="hint">No hay datos de circuito para mostrar.</div>';
      }

      const nodes = [...circuit.nodes].sort((a, b) => {
        if ((a.level || 0) !== (b.level || 0)) {
          return (a.level || 0) - (b.level || 0);
        }
        return String(a.label || '').localeCompare(String(b.label || ''));
      });

      const levels = {};
      for (const node of nodes) {
        const level = Number.isFinite(node.level) ? node.level : 0;
        if (!levels[level]) levels[level] = [];
        levels[level].push(node);
      }

      const levelKeys = Object.keys(levels).map((value) => Number(value)).sort((a, b) => a - b);
      const maxLevel = levelKeys.length ? levelKeys[levelKeys.length - 1] : 0;
      const maxNodesInLevel = levelKeys.reduce((max, level) => Math.max(max, levels[level].length), 1);

      const slotWidth = 112;
      const slotHeight = 78;
      const xSpacing = 170;
      const ySpacing = 94;
      const marginX = 48;
      const marginY = 32;

      const width = marginX * 2 + (maxLevel * xSpacing) + slotWidth + 72;
      const height = marginY * 2 + ((maxNodesInLevel - 1) * ySpacing) + slotHeight + 30;

      const slotByNode = {};
      for (const level of levelKeys) {
        const nodesAtLevel = levels[level];
        const blockHeight = (nodesAtLevel.length - 1) * ySpacing;
        const startY = ((height - slotHeight) - blockHeight) / 2;
        nodesAtLevel.forEach((node, index) => {
          slotByNode[node.id] = {
            x: marginX + (level * xSpacing),
            y: startY + (index * ySpacing),
          };
        });
      }

      const nodeById = {};
      for (const node of nodes) {
        nodeById[node.id] = node;
      }

      const anchors = {};
      const nodeElements = nodes.map((node) => {
        const slot = slotByNode[node.id];
        if (!slot) return '';

        const box = getNodeBox(node);
        const x = slot.x + (slotWidth - box.width) / 2;
        const y = slot.y + (slotHeight - box.height) / 2;
        const cx = x + box.width / 2;
        const cy = y + box.height / 2;
        const nodeType = String(node.type || '').toLowerCase();
        const gateType = resolveCircuitGateType(node);
        let markup = '';
        let leftX = x;
        let rightX = x + box.width;

        if (box.io) {
          const radius = box.width / 2;
          const ioClass = nodeType === 'input'
            ? 'circuit-input'
            : (nodeType === 'output' ? 'circuit-output' : 'circuit-const');
          markup = [
            `<circle class="circuit-io ${ioClass}" cx="${cx}" cy="${cy}" r="${radius}"></circle>`,
            `<text class="circuit-label" x="${cx}" y="${cy + 4}" text-anchor="middle">${escapeHtml(String(node.label || ''))}</text>`,
          ].join('');
          leftX = cx - radius;
          rightX = cx + radius;
          anchors[node.id] = {
            left: { x: leftX, y: cy },
            right: { x: rightX, y: cy },
            body: { top: y, height: box.height },
            isGate: !box.io,
            inputBoundary: { kind: 'line', x: leftX },
          };
        } else {
          const gateSvg = drawGateSymbol(gateType, x, y, box.width, box.height, String(node.label || ''));
          markup = gateSvg.markup;
          leftX = gateSvg.leftX;
          rightX = gateSvg.rightX;
          anchors[node.id] = {
            left: { x: leftX, y: cy },
            right: { x: rightX, y: cy },
            body: { top: y, height: box.height },
            isGate: !box.io,
            inputBoundary: gateSvg.inputBoundary || { kind: 'line', x: leftX },
          };
        }

        return [
          `<g data-node="${escapeHtml(node.id)}">`,
          markup,
          `<title>${escapeHtml(String(node.label || ''))} [${escapeHtml(String(node.type || 'node'))}]</title>`,
          '</g>',
        ].join('');
      }).join('');

      const edges = circuit.edges || [];
      const incomingByTarget = {};
      edges.forEach((edge, index) => {
        if (!incomingByTarget[edge.to]) incomingByTarget[edge.to] = [];
        const sourceY = anchors[edge.from]?.right?.y ?? 0;
        incomingByTarget[edge.to].push({ index, sourceY });
      });

      const edgeTargetYByIndex = {};
      Object.entries(incomingByTarget).forEach(([targetId, entries]) => {
        const targetAnchor = anchors[targetId];
        if (!targetAnchor) return;

        const targetNode = nodeById[targetId];
        const targetType = String(targetNode?.type || '').toLowerCase();
        const shouldSpreadInputs = targetAnchor.isGate && targetType !== 'output' && entries.length > 1;

        if (!shouldSpreadInputs) {
          for (const entry of entries) {
            edgeTargetYByIndex[entry.index] = targetAnchor.left.y;
          }
          return;
        }

        const sorted = [...entries].sort((a, b) => a.sourceY - b.sourceY);
        sorted.forEach((entry, portIndex) => {
          const y = targetAnchor.body.top + ((portIndex + 1) * targetAnchor.body.height) / (sorted.length + 1);
          edgeTargetYByIndex[entry.index] = y;
        });
      });

      function getInputBoundaryX(targetAnchor, targetY) {
        const boundary = targetAnchor?.inputBoundary || { kind: 'line', x: targetAnchor?.left?.x ?? 0 };
        if (boundary.kind === 'quadratic') {
          const h = Math.max(1, Number(boundary.height || 1));
          const top = Number(boundary.top || 0);
          const tRaw = (targetY - top) / h;
          const t = Math.max(0, Math.min(1, tRaw));
          const omt = 1 - t;
          const xStart = Number(boundary.xStart || 0);
          const xCtrl = Number(boundary.xCtrl || xStart);
          return (omt * omt * xStart) + (2 * omt * t * xCtrl) + (t * t * xStart);
        }
        return Number(boundary.x || targetAnchor?.left?.x || 0);
      }

      const edgeElements = edges.map((edge, index) => {
        const from = anchors[edge.from]?.right;
        const targetAnchor = anchors[edge.to];
        const toLeft = targetAnchor?.left;
        const to = toLeft
          ? {
            x: getInputBoundaryX(targetAnchor, edgeTargetYByIndex[index] ?? toLeft.y),
            y: edgeTargetYByIndex[index] ?? toLeft.y,
          }
          : null;
        if (!from || !to) return '';

        const delta = Math.max(28, (to.x - from.x) * 0.35);
        const ctrlX1 = from.x + delta;
        const ctrlX2 = to.x - delta;
        return `<path class="circuit-wire" d="M ${from.x} ${from.y} C ${ctrlX1} ${from.y}, ${ctrlX2} ${to.y}, ${to.x} ${to.y}"></path>`;
      }).join('');

      return [
        '<div class="circuit-wrap">',
        `<svg class="circuit-svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(circuit.title || 'Circuito lógico')}">`,
        edgeElements,
        nodeElements,
        '</svg>',
        '</div>',
      ].join('');
    }

    function renderCircuitGraph(circuitKey) {
      if (!latestCircuits) return;
      if (!latestCircuits[circuitKey]) {
        const available = Object.keys(CIRCUIT_OPTIONS).find((key) => latestCircuits[key]);
        if (!available) return;
        activeCircuitKey = available;
      } else {
        activeCircuitKey = circuitKey;
      }

      const canvas = document.getElementById('circuitCanvas');
      if (!canvas) return;
      canvas.innerHTML = buildCircuitSvg(latestCircuits[activeCircuitKey]);

      document.querySelectorAll('input[name="circuitType"]').forEach((input) => {
        const isActive = input.value === activeCircuitKey;
        input.checked = isActive;
        const label = input.closest('.circuit-radio');
        if (label) {
          label.classList.toggle('active', isActive);
        }
      });
    }

    function renderCircuitPanel(circuits) {
      const availableKeys = Object.keys(CIRCUIT_OPTIONS).filter((key) => circuits && circuits[key]);
      if (availableKeys.length === 0) {
        latestCircuits = null;
        circuitPanel.classList.add('hidden');
        circuitPanel.innerHTML = '';
        return;
      }

      latestCircuits = circuits;
      if (!availableKeys.includes(activeCircuitKey)) {
        activeCircuitKey = availableKeys[0];
      }

      const radiosHtml = availableKeys
        .map((key) => {
          const checked = key === activeCircuitKey ? 'checked' : '';
          const activeClass = key === activeCircuitKey ? ' active' : '';
          return [
            `<label class="circuit-radio${activeClass}">`,
            `<input type="radio" name="circuitType" value="${escapeHtml(key)}" ${checked}>`,
            `<span>${escapeHtml(CIRCUIT_OPTIONS[key])}</span>`,
            '</label>',
          ].join('');
        })
        .join('');

      circuitPanel.classList.remove('hidden');
      circuitPanel.innerHTML = [
        '<details class="ondemand plain" id="circuitDetails">',
        '<summary>Diagrama del circuito</summary>',
        '<fieldset class="circuit-radio-group">',
        radiosHtml,
        '</fieldset>',
        '<div id="circuitCanvas"></div>',
        '</details>',
      ].join('');

      document.querySelectorAll('input[name="circuitType"]').forEach((input) => {
        input.addEventListener('change', () => {
          if (input.checked) {
            renderCircuitGraph(input.value);
          }
        });
      });

      const circuitDetails = document.getElementById('circuitDetails');
      if (circuitDetails) {
        circuitDetails.addEventListener('toggle', () => {
          if (circuitDetails.open) {
            renderCircuitGraph(activeCircuitKey);
          }
        });
      }
    }

    function renderGatePanel(gates) {
      const aon = gates?.and_or_not || { and: 0, or: 0, not: 0, total: 0 };
      const nand = gates?.nand_only || { gate_count: 0, output_signal: '', netlist: '', notes: [] };
      const nor = gates?.nor_only || { gate_count: 0, output_signal: '', netlist: '', notes: [] };

      function renderNetwork(title, network) {
        const notes = (network.notes || [])
          .map((note) => `<div class="hint">${escapeHtml(note)}</div>`)
          .join('');

        return [
          `<div><strong>${title}:</strong> ${network.gate_count ?? 0} compuertas</div>`,
          `<div><strong>Salida:</strong> <code>${escapeHtml(network.output_signal || '')}</code></div>`,
          network.netlist ? `<pre>${escapeHtml(network.netlist)}</pre>` : '',
          notes,
        ].join('');
      }

      gatePanel.classList.remove('hidden');
      gatePanel.innerHTML = [
        '<h2 class="synthesis-title">Síntesis con compuertas</h2>',
        `<div><strong>AND/OR/NOT:</strong> AND=${aon.and}, OR=${aon.or}, NOT=${aon.not}, Total=${aon.total}</div>`,
        '<hr>',
        renderNetwork('Solo NAND', nand),
        '<hr>',
        renderNetwork('Solo NOR', nor),
      ].join('');
    }

    function renderTruthPanel(truthTable) {
      const rows = truthTable?.rows || [];
      const header = truthTable?.header || [];
      const previewLimit = 64;
      const previewRows = rows.slice(0, previewLimit);
      const truncated = rows.length > previewRows.length;

      const headHtml = `<tr>${header.map((name) => `<th>${escapeHtml(name)}</th>`).join('')}</tr>`;
      const bodyHtml = previewRows
        .map((row) => {
          const values = [...(row.bits || []), row.output];
          return `<tr>${values.map((value) => `<td>${escapeHtml(value)}</td>`).join('')}</tr>`;
        })
        .join('');

      truthPanel.classList.remove('hidden');
      truthPanel.innerHTML = [
        '<details class="ondemand plain" id="truthDetails">',
        '<summary>Tabla de verdad</summary>',
        '<div id="truthContent"></div>',
        '</details>',
      ].join('');

      const truthDetails = document.getElementById('truthDetails');
      const truthContent = document.getElementById('truthContent');
      if (!truthDetails || !truthContent) return;

      truthDetails.addEventListener('toggle', () => {
        if (!truthDetails.open || truthContent.dataset.rendered === '1') return;
        truthContent.innerHTML = [
          `<div class="hint">Filas mostradas: ${previewRows.length}${truncated ? ` de ${rows.length}` : ''}</div>`,
          '<div class="mini-table truth-table-gap">',
          `<table><thead>${headHtml}</thead><tbody>${bodyHtml}</tbody></table>`,
          '</div>',
        ].join('');
        truthContent.dataset.rendered = '1';
      });
    }

    function renderSimplifyResult(data) {
      const labels = {
        sop: 'Suma de productos (SOP)',
        pos: 'Producto de sumas (POS)',
      };
      const terms = data.terms || { minterms: [], maxterms: [], dontcares: [] };
      const eq = data.equivalence || {};

      const eqLine = eq.equivalent
        ? `Sí (${eq.method || 'auto'}, casos revisados: ${eq.checked_cases || 0}${eq.guaranteed ? ', verificación completa' : ''})`
        : `No (${eq.method || 'auto'})`;

      const counterExampleHtml = eq.counterexample
        ? `<div class="result-counterexample"><strong>Contraejemplo:</strong> índice ${escapeHtml(eq.counterexample.index)} | A=${escapeHtml(eq.counterexample.value_a)} | B=${escapeHtml(eq.counterexample.value_b)} | asignación ${escapeHtml(JSON.stringify(eq.counterexample.assignment))}</div>`
        : '';

      const variablesText = (data.variables || []).length
        ? escapeHtml(data.variables.join(', '))
        : 'Ninguna (constante)';
      const proofClass = eq.equivalent ? 'result-proof' : 'result-proof failed';

      resultBox.classList.remove('hidden', 'error');
      resultBox.classList.add('ok');
      resultBox.innerHTML = [
        '<div class="result-head">',
        '<div>',
        '<span class="result-kicker">Resultado minimizado</span>',
        `<div class="result-format">${labels[data.result_form] || escapeHtml(data.result_form)}</div>`,
        '</div>',
        `<code class="result-expression">${escapeHtml(data.simplified_expression || '')}</code>`,
        '</div>',
        '<div class="result-grid">',
        `<div class="result-item"><span>Variables</span><strong>${variablesText}</strong></div>`,
        `<div class="result-item"><span>Expresión base</span><strong><code>${escapeHtml(data.expression || '')}</code></strong></div>`,
        `<div class="result-item"><span>Casos revisados</span><strong>${escapeHtml(eq.checked_cases || 0)}</strong></div>`,
        `<div class="result-item"><span>Minitérminos</span><strong>${escapeHtml(formatList(terms.minterms))}</strong></div>`,
        `<div class="result-item"><span>Maxitérminos</span><strong>${escapeHtml(formatList(terms.maxterms))}</strong></div>`,
        `<div class="result-item"><span>Don't cares</span><strong>${escapeHtml(formatList(terms.dontcares))}</strong></div>`,
        '</div>',
        `<div class="${proofClass}">Equivalencia comprobada: ${escapeHtml(eqLine)}</div>`,
        counterExampleHtml,
      ].join('');

      renderTruthPanel(data.truth_table);
      renderGatePanel(data.gate_synthesis);
      renderCircuitPanel(data.circuits);
    }

    function renderSimplifyError(message) {
      resultBox.classList.remove('hidden', 'ok');
      resultBox.classList.add('error');
      resultBox.textContent = message;
      latestCircuits = null;

      [gatePanel, circuitPanel, truthPanel].forEach((panel) => {
        panel.classList.add('hidden');
        panel.innerHTML = '';
      });
    }

    document.getElementById('logicForm').addEventListener('submit', async (event) => {
      event.preventDefault();

      const payload = {
        mode: document.getElementById('mode').value,
        result_form: document.getElementById('result_form').value,
        variables: document.getElementById('variables').value,
        equation: document.getElementById('equation').value,
        table: document.getElementById('table').value,
        terms: document.getElementById('terms').value,
        dontcares: document.getElementById('dontcares').value,
      };

      renderSimplifyError('Procesando...');

      try {
        const response = await fetch('/api/simplify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await response.json();

        if (!response.ok || !data.ok) {
          throw new Error(data.error || 'No se pudo simplificar la función.');
        }

        renderSimplifyResult(data);
      } catch (error) {
        renderSimplifyError(error.message || 'Error en simplificación.');
      }
    });

    document.getElementById('equivalenceForm').addEventListener('submit', async (event) => {
      event.preventDefault();

      equivalenceResultBox.classList.remove('hidden', 'ok');
      equivalenceResultBox.classList.add('error');
      equivalenceResultBox.textContent = 'Verificando...';

      const payload = {
        expression_a: document.getElementById('expressionA').value,
        expression_b: document.getElementById('expressionB').value,
        variables: document.getElementById('equivVariables').value,
        method: document.getElementById('equivMethod').value,
        sample_size: document.getElementById('equivSample').value,
      };

      try {
        const response = await fetch('/api/equivalence', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await response.json();

        if (!response.ok || !data.ok) {
          throw new Error(data.error || 'No se pudo verificar equivalencia.');
        }

        const result = data.result || {};
        const verdict = result.equivalent ? 'Equivalentes' : 'No equivalentes';
        const guarantee = result.guaranteed ? 'verificación completa' : 'verificación por muestra';

        let counterexample = '';
        if (result.counterexample) {
          counterexample = `<div><strong>Contraejemplo:</strong> índice ${escapeHtml(result.counterexample.index)} | A=${escapeHtml(result.counterexample.value_a)} | B=${escapeHtml(result.counterexample.value_b)} | asignación ${escapeHtml(JSON.stringify(result.counterexample.assignment))}</div>`;
        }

        equivalenceResultBox.classList.remove('error');
        equivalenceResultBox.classList.add('ok');
        equivalenceResultBox.innerHTML = [
          `<div><strong>Resultado:</strong> ${verdict}</div>`,
          `<div><strong>Método:</strong> ${escapeHtml(result.method || 'auto')} (${escapeHtml(guarantee)})</div>`,
          `<div><strong>Casos revisados:</strong> ${escapeHtml(result.checked_cases || 0)} / ${escapeHtml(result.total_care_cases || 0)}</div>`,
          `<div><strong>Variables:</strong> ${escapeHtml((data.variables || []).join(', ') || '(ninguna)')}</div>`,
          `<div><strong>Expresión 1:</strong> <code>${escapeHtml(data.expression_a || '')}</code></div>`,
          `<div><strong>Expresión 2:</strong> <code>${escapeHtml(data.expression_b || '')}</code></div>`,
          counterexample,
        ].join('');
      } catch (error) {
        equivalenceResultBox.classList.remove('ok');
        equivalenceResultBox.classList.add('error');
        equivalenceResultBox.textContent = error.message || 'Error en equivalencia.';
      }
    });

    document.getElementById('convertForm').addEventListener('submit', async (event) => {
      event.preventDefault();

      convertResultBox.classList.remove('hidden', 'ok');
      convertResultBox.classList.add('error');
      convertResultBox.textContent = 'Convirtiendo...';

      const direction = document.getElementById('convertDirection').value;
      const payload = {
        direction,
        variables: document.getElementById('convertVariables').value,
        equation: document.getElementById('convertEquation').value,
        table: document.getElementById('convertTable').value,
        dontcares: document.getElementById('convertDontcares').value,
      };

      try {
        const response = await fetch('/api/convert', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await response.json();

        if (!response.ok || !data.ok) {
          throw new Error(data.error || 'No se pudo convertir.');
        }

        convertResultBox.classList.remove('error');
        convertResultBox.classList.add('ok');

        if (direction === 'equation_to_table') {
          convertResultBox.innerHTML = [
            `<div><strong>Variables:</strong> ${escapeHtml((data.variables || []).join(', ') || '(ninguna)')}</div>`,
            `<div><strong>SOP:</strong> <code>${escapeHtml(data.sop_expression || '')}</code></div>`,
            `<div><strong>POS:</strong> <code>${escapeHtml(data.pos_expression || '')}</code></div>`,
            '<div><strong>Tabla generada (CSV):</strong></div>',
            `<pre>${escapeHtml(data.truth_table?.csv || '')}</pre>`,
            '<div class="actions">',
            '<button type="button" class="secondary" id="useGeneratedTable">Usar tabla en simplificador</button>',
            '</div>',
          ].join('');

          const useTableButton = document.getElementById('useGeneratedTable');
          if (useTableButton) {
            useTableButton.onclick = () => {
              modeElement.value = 'table';
              refreshVisibleFields();
              document.getElementById('variables').value = (data.variables || []).join(', ');
              document.getElementById('table').value = data.truth_table?.csv || '';
            };
          }
        } else {
          convertResultBox.innerHTML = [
            `<div><strong>Variables:</strong> ${escapeHtml((data.variables || []).join(', ') || '(ninguna)')}</div>`,
            `<div><strong>SOP:</strong> <code>${escapeHtml(data.sop_expression || '')}</code></div>`,
            `<div><strong>POS:</strong> <code>${escapeHtml(data.pos_expression || '')}</code></div>`,
            '<div class="actions">',
            '<button type="button" class="secondary" id="useSopEquation">Usar SOP en simplificador</button>',
            '<button type="button" class="secondary" id="usePosEquation">Usar POS en simplificador</button>',
            '</div>',
          ].join('');

          const useSopButton = document.getElementById('useSopEquation');
          if (useSopButton) {
            useSopButton.onclick = () => {
              modeElement.value = 'equation';
              refreshVisibleFields();
              document.getElementById('variables').value = (data.variables || []).join(', ');
              document.getElementById('equation').value = data.sop_expression || '';
              document.getElementById('result_form').value = 'sop';
            };
          }

          const usePosButton = document.getElementById('usePosEquation');
          if (usePosButton) {
            usePosButton.onclick = () => {
              modeElement.value = 'equation';
              refreshVisibleFields();
              document.getElementById('variables').value = (data.variables || []).join(', ');
              document.getElementById('equation').value = data.pos_expression || '';
              document.getElementById('result_form').value = 'pos';
            };
          }
        }
      } catch (error) {
        convertResultBox.classList.remove('ok');
        convertResultBox.classList.add('error');
        convertResultBox.textContent = error.message || 'Error en conversión.';
      }
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
