# Construye el Excel maestro: las 6 tablas reales de Supabase (leídas en
# exportar_supabase.py, sin tocar producción) + limpieza/completado de
# `proveedores.valoracion` con trazabilidad explícita de qué valores son
# reales (Google Places) y cuáles son un valor por defecto imputado.
#
# Estrategia de relleno para valoración sin dato real (ver conversación):
# se PRESERVA la convención que ya existe en el dataset/producción (no se
# inventa una nueva): 3.0 para proveedores nunca buscados en Google Places,
# 4.0 para proveedores buscados pero sin resultado de Google (no encontrado
# o encontrado sin reseñas). Se documenta como hallazgo en docs/DECISIONES.md
# porque contradice lo escrito en el TFM (que documenta un único default de
# 3.0) -- no se resuelve esa discrepancia acá, solo se hace explícita.
import numpy as np
import pandas as pd

RUTA_ORIGINALES = "../../medi-match-db/dataset/_originales_sin_procesar"
RUTA_SALIDA = "../../medi-match-db/dataset/medimatch_supabase_completo.xlsx"

# --- 1. Tablas reales de Supabase (ya descargadas por exportar_supabase.py) ---
proveedores = pd.read_pickle("_export_supabase_raw/proveedores.pkl")
costo_tratamientos = pd.read_pickle("_export_supabase_raw/costo_tratamientos.pkl")
capacidades = pd.read_pickle("_export_supabase_raw/capacidades.pkl")
clientes = pd.read_pickle("_export_supabase_raw/clientes.pkl")
gestores = pd.read_pickle("_export_supabase_raw/gestores.pkl")
asignaciones = pd.read_pickle("_export_supabase_raw/asignaciones.pkl")

# --- 2. Reconstruir el estado de Google Places por proveedor (mismo análisis
#     de la conversación: ok / no_intentado / no_encontrado / encontrado_sin_rating) ---
gp = pd.read_excel(f"{RUTA_ORIGINALES}/google_places_proveedorV2_20260812_010426_1.xlsx", sheet_name="Resumen")
gp_enriquecido = pd.read_excel(f"{RUTA_ORIGINALES}/google_places_proveedorV2_20260812_010426_1_enriquecido.xlsx", sheet_name="Resumen")

proveedores["NOMBRE_NORM"] = proveedores["nombre_proveedor"].str.upper().str.strip()
gp["NOMBRE_NORM"] = gp["NOMBRE_CENTRO"].str.upper().str.strip()
gp_enriquecido["NOMBRE_NORM"] = gp_enriquecido["NOMBRE_CENTRO"].str.upper().str.strip()

cols_gp = gp[["NOMBRE_NORM", "encontrado", "rating_global", "numero_valoraciones", "place_id", "google_maps_url"]].rename(
    columns={"rating_global": "rating_google_places", "numero_valoraciones": "numero_valoraciones_google"}
)
cols_candidato = gp_enriquecido[
    ["NOMBRE_NORM", "nombre_comercial_candidato", "tipo_candidato", "confianza_candidato", "nota_candidato"]
]

proveedores = proveedores.merge(cols_gp, on="NOMBRE_NORM", how="left").merge(cols_candidato, on="NOMBRE_NORM", how="left")


def clasificar(fila):
    if pd.isna(fila["encontrado"]):
        return "no_intentado"
    if not fila["encontrado"]:
        return "no_encontrado"
    if pd.isna(fila["rating_google_places"]):
        return "encontrado_sin_rating"
    return "ok"


proveedores["estado_google_places"] = proveedores.apply(clasificar, axis=1)

# --- 3. Trazabilidad de la valoración: fuente + flag de imputación ---
FUENTE_POR_ESTADO = {
    "ok": "google_maps",
    "no_intentado": "default_no_intentado_3.0",
    "no_encontrado": "default_intentado_sin_resultado_4.0",
    "encontrado_sin_rating": "default_intentado_sin_resultado_4.0",
}
proveedores["fuente_valoracion"] = proveedores["estado_google_places"].map(FUENTE_POR_ESTADO)
proveedores["valoracion_imputada"] = proveedores["fuente_valoracion"] != "google_maps"

# Verificación: el valor ya cargado en producción (`valoracion`) debe ser
# consistente con la convención reconstruida -- si no, es una alerta real,
# no un error de este script.
esperado = proveedores["estado_google_places"].map({"no_intentado": 3.0, "no_encontrado": 4.0, "encontrado_sin_rating": 4.0})
inconsistentes = proveedores[proveedores["valoracion_imputada"] & (proveedores["valoracion"] != esperado)]
print(f"proveedores con valoración imputada (no viene de Google): {proveedores['valoracion_imputada'].sum()} / {len(proveedores)}")
print(f"inconsistentes con la convención esperada (revisar a mano): {len(inconsistentes)}")

proveedores = proveedores.drop(columns=["NOMBRE_NORM"])

# --- 4. Capacidad estandarizada a semanas (ver cuaderno de ranking) ---
capacidades["capacidad_restante_semanal"] = capacidades["capacidad_restante"] / 52
capacidades["capacidad_maxima_semanal"] = capacidades["capacidad_maxima"] / 52

# --- 5. Hoja de metadatos/README ---
metadata = pd.DataFrame(
    {
        "hoja": [
            "proveedores", "costo_tratamientos", "capacidades", "clientes", "gestores", "asignaciones",
        ],
        "descripcion": [
            "1 fila por clínica real. Incluye valoracion (la que usa producción hoy) + columnas nuevas "
            "de trazabilidad: rating_google_places/numero_valoraciones_google (dato real cuando existe), "
            "estado_google_places, fuente_valoracion, valoracion_imputada (True = no viene de Google), y "
            "nombre_comercial_candidato/confianza_candidato para los casos sin resultado (ver hoja README).",
            "Catálogo proveedor+tratamiento: coste_medio, pasar_modelo (activa el motor de ranking).",
            "Capacidad por (proveedor, tratamiento). Se agregan capacidad_restante_semanal/"
            "capacidad_maxima_semanal (capacidad/52) para comparar contra demanda pronosticada semanal.",
            "Clientes de prueba (datos sensibles: póliza, documento, nombre).",
            "Cuentas de gestores (login). password_hash es bcrypt, irreversible.",
            "Reservas activas/eliminadas (soft delete vía columna estado).",
        ],
        "filas": [len(proveedores), len(costo_tratamientos), len(capacidades), len(clientes), len(gestores), len(asignaciones)],
    }
)

nota_valoracion = pd.DataFrame(
    {
        "": [
            "ESTRATEGIA DE RELLENO DE VALORACIÓN (proveedores.valoracion)",
            "",
            "No se inventó un valor nuevo -- se preservó y se documentó la convención que YA existe",
            "en el dataset/producción (verificado 1:1 contra Supabase real, 2026-09-10):",
            "",
            "  - fuente_valoracion = 'google_maps': rating real de Google Places (178/297 proveedores).",
            "  - fuente_valoracion = 'default_no_intentado_3.0': nunca se buscó en Google Places (81/297).",
            "  - fuente_valoracion = 'default_intentado_sin_resultado_4.0': se buscó pero Google no devolvió",
            "    rating -- ya sea 'no encontrado' o 'encontrado sin reseñas' (38/297).",
            "",
            "HALLAZGO (ver docs/DECISIONES.md, entrada 2026-09-10): el TFM (Grupo_3_TFM..., cap. 3.5)",
            "documenta un ÚNICO default de 3.0 para 'sin valoración conocida'. Los datos reales usan DOS",
            "defaults distintos según la causa (3.0 vs 4.0). Se deja explícito con esta columna para que",
            "se decida a propósito, no que quede oculto dentro de un número que parece un dato real.",
            "",
            "Para los 'no_encontrado' con nombre_comercial_candidato de confianza alta/media (columna en",
            "la hoja proveedores), reintentar el mismo pipeline de Google Places con ese nombre antes de",
            "seguir usando el default -- son los que más probablemente sí tengan un rating real disponible.",
        ]
    }
)

with pd.ExcelWriter(RUTA_SALIDA, engine="openpyxl") as writer:
    metadata.to_excel(writer, sheet_name="README", index=False, startrow=0)
    nota_valoracion.to_excel(writer, sheet_name="README", index=False, header=False, startrow=len(metadata) + 3)
    proveedores.to_excel(writer, sheet_name="proveedores", index=False)
    costo_tratamientos.to_excel(writer, sheet_name="costo_tratamientos", index=False)
    capacidades.to_excel(writer, sheet_name="capacidades", index=False)
    clientes.to_excel(writer, sheet_name="clientes", index=False)
    gestores.to_excel(writer, sheet_name="gestores", index=False)
    asignaciones.to_excel(writer, sheet_name="asignaciones", index=False)

print(f"\nEscrito: {RUTA_SALIDA}")
print(f"\nDistribución fuente_valoracion:")
print(proveedores["fuente_valoracion"].value_counts())
