# Simplificador Booleano (Python + Vercel)

Aplicación web en Python para simplificar funciones booleanas con `sympy.logic`.

## Funcionalidades

- Entrada por **ecuación**, **tabla de verdad**, **minitérminos** o **maxitérminos**.
- Simplificación en **SOP (Suma de Productos)** o **POS (Producto de Sumas)**.
- **Diagrama del circuito lógico** de la expresión simplificada (vista AND/OR/NOT, NAND y NOR).
- **Verificador de equivalencia** entre dos expresiones (modo exhaustivo o muestreo).
- **Conversión automática**:
  - Ecuación -> tabla de verdad
  - Tabla de verdad -> ecuaciones SOP/POS
- **Exportación** de resultados a **JSON**, **CSV (tabla/términos)** y **LaTeX**.
- **Síntesis de compuertas** con conteo:
  - Base AND/OR/NOT
  - Generación **solo NAND** (netlist + conteo exacto)
  - Generación **solo NOR** (netlist + conteo exacto)

## Estructura

- `api/index.py`: app Flask, endpoints y UI.
- `requirements.txt`: dependencias Python.
- `vercel.json`: configuración de despliegue en Vercel.

## Ejecutar local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 api/index.py
```

Abrir en navegador: `http://localhost:5000`

Si `5000` está ocupada:

```bash
flask --app api/index.py run --port 5001
```

## Desplegar en Vercel

```bash
npm i -g vercel
vercel login
vercel
```

La ruta principal y todos los endpoints son servidos por `api/index.py`.

## Endpoints

### `POST /api/simplify`

Simplifica una función booleana desde cualquiera de los modos de entrada.

Campos principales:

- `mode`: `equation` | `table` | `minterms` | `maxterms`
- `result_form`: `sop` | `pos`
- `variables`: lista separada por comas
- `equation`: texto de ecuación
- `table`: texto de tabla
- `terms`: índices de minitérminos o maxitérminos
- `dontcares`: índices don't care

Respuesta incluye además de la simplificación:

- `truth_table`
- `equivalence`
- `gate_synthesis`
- `gate_synthesis.nand_only.netlist`
- `gate_synthesis.nor_only.netlist`
- `circuits` (grafos para dibujar AON/NAND/NOR)
- `exports` (CSV/LaTeX)

### `POST /api/equivalence`

Verifica equivalencia entre dos expresiones.

Campos:

- `expression_a`
- `expression_b`
- `variables` (opcional)
- `method`: `auto` | `exhaustive` | `sample`
- `sample_size`

### `POST /api/convert`

Conversión entre ecuación y tabla.

Campos:

- `direction`: `equation_to_table` | `table_to_equation`
- `variables` (opcional)
- `equation` (si aplica)
- `table` (si aplica)
- `dontcares` (opcional)

## Notas de ecuaciones

Operadores admitidos:

- `~` o `!` para NOT
- `&` o `*` para AND
- `|` o `+` para OR
- `^` para XOR
- también palabras `AND`, `OR`, `NOT`, `XOR`
