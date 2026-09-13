# Enriquece (no sobreescribe) el archivo de Google Places con candidatos de
# nombre comercial para los proveedores "no encontrado", extraídos de
# DIRECCION_ORIGINAL y verificados con búsquedas web puntuales (ver
# conversación). No inventa rating_global/numero_valoraciones -- eso requiere
# volver a correr el mismo pipeline de Google Places con estos candidatos.
import pandas as pd

RUTA_ORIGINAL = "../../medi-match-db/dataset/google_places_proveedorV2_20260812_010426_1.xlsx"
RUTA_SALIDA = "../../medi-match-db/dataset/google_places_proveedorV2_20260812_010426_1_enriquecido.xlsx"

# Candidato por ID -- construido a mano revisando cada DIRECCION_ORIGINAL,
# con verificación web para los de confianza "alta" (ver conversación: Timalex
# -> Oncomedica, Sanatrix, Fenix Salud, Cemo/Cemanex y Torre Maracaibo se
# confirmaron como lugares reales y existentes).
CANDIDATOS = {
    "V-12898300-0": ("Edificio Majestic, Av Libertador", "edificio", "baja",
                      "Dirección original viene corrupta (\".Libertador .Majestic .6\"); candidato tentativo, sin verificar. Búsqueda del nombre propio no dio resultados."),
    "V-14573821-0": (None, None, None, "Dirección es solo calle + zona genérica (Casco Central), sin nombre de edificio/clínica que buscar."),
    "V-16291543-0": (None, None, None, "Dirección es un cruce de avenidas, sin nombre de edificio/clínica."),
    "J-41266255-4": ("IPS Carrizal, Edificio Oficentro, Los Teques", "edificio", "media",
                      "Razón social completa (\"...INVERSIONES DANISLAUH C.A.\") probablemente no es el nombre comercial -- probar solo \"IPS Carrizal\" + Los Teques."),
    "V-6693443-0": (None, None, None, "Edificio de RESIDENCIAS (Los Girasoles) -- parece consultorio en apartamento, no un centro médico con ficha propia."),
    "V-14203128-0": ("Tamasalud CCCT", "clinica_o_centro_medico", "media",
                      "\"Tamasalud\" es una cadena de centros médicos conocida en Venezuela; coincide con \"Chuao Ccct... Tamasalud\" de la dirección original. No verificado en esta sesión con una búsqueda dedicada, pero es un nombre de marca real, no genérico."),
    "J-30166120-6": ("SERMECA", "clinica_o_centro_medico", "media",
                      "Acrónimo al final de la razón social (\"SERVICIOS MEDICOS ASISTENCIALES C.A SERMECA\") -- probar ese nombre corto en vez de la razón social completa."),
    "V-5891026-0": ("Clínica CEMO / CEMANEX", "clinica_o_centro_medico", "alta",
                     "Verificado por búsqueda web: \"CEMANEX S.C. Anexo de Consultorios Clínica CEMO, Calle Cristóbal Rojas\" coincide EXACTO con la dirección original. Clínica CEMO es un centro real en Av. Arturo Michelena, Santa Mónica, Caracas."),
    "V-2767349-0": ("Torre Maracaibo, Av. Libertador", "edificio", "alta",
                     "Verificado por búsqueda web: \"Torre Maracaibo\" es un edificio real de consultorios médicos en Av. Libertador, Caracas (frente a la Clínica Santiago de León) -- coincide con \"Torre Maracaibo Piso 9\" de la dirección original."),
    "V-10173301-0": ("Torre Maracaibo, Av. Libertador", "edificio", "alta",
                      "Mismo edificio verificado que el caso anterior (Piso 11 en vez de Piso 9)."),
    "V-10111488-0": ("Centro Clínico Fénix Salud, San Bernardino", "clinica_o_centro_medico", "alta",
                      "Verificado por búsqueda web: \"Fenix Salud\" = Centro Clínico Fénix Salud, Av. Guaicaipuro, San Bernardino -- clínica real con más de 70 consultorios, coincide con la dirección original."),
    "V-6139617-0": ("Torre Auyantepuy, Av. Auyantepuy", "edificio", "baja",
                     "\"Auyantepuy Commodoro 3\" sugiere un edificio residencial/de oficinas, no necesariamente un centro médico -- candidato débil, no verificado."),
    "V-6824942-0": (None, None, None, "Dirección es un cruce de avenidas, sin nombre de edificio/clínica."),
    "V-5577613-0": ("Edificio San Carlos, Av. Araure", "edificio", "baja",
                     "\"Apto 2\" sugiere apartamento residencial, no consultorio -- candidato débil."),
    "V-12671607-0": (None, None, None, "Dirección es un sector/urbanización (Terrazas del Ávila), sin nombre de edificio/clínica."),
    "J-31763269-9": ("Oncomedica (nombre comercial de Corporación Médica Timalex)", "clinica_o_centro_medico", "alta",
                      "Verificado por búsqueda web: Corporación Médica Timalex C.A. opera comercialmente como \"Oncomedica\" (centro de oncología/quimioterapia), Edif. Maracaibo Piso 9, Av. Libertador -- la búsqueda automática solo probó la razón social legal."),
    "V-19857704-0": (None, None, None, "Dirección truncada/residencial (\"casa nro 107.14\"), sin nombre de edificio/clínica."),
    "V-3186793-0": ("Clínica Sanatrix", "clinica_o_centro_medico", "alta",
                     "Verificado por búsqueda web: Clínica Sanatrix es un centro médico real en Campo Alegre/Chacao, Caracas -- coincide con \"Edif. Clinica Sanatrix\" de la dirección original."),
    "V-6374676-0": (None, None, None, "\"Apt 11\" en un edificio residencial (Marare) -- no parece tener ficha propia de centro médico."),
    "V-8008743-0": (None, None, None, "Dirección es un cruce de avenidas, sin nombre de edificio/clínica."),
    "V-10670501-0": ("Edificio Libertador 75 (frente a Policlínica Santiago de León)", "edificio", "media",
                      "La Policlínica Santiago de León es una referencia (landmark) cercana, no el lugar de trabajo del proveedor -- probar el nombre del edificio propio, no el landmark."),
    "V-17427900-0": (None, None, None, "Dirección residencial (\"Casa #37\"), sin nombre de edificio/clínica."),
    "J-9019064-3": ("Instituto de Cirugía Ambulatoria, Barrio Obrero, San Cristóbal", "clinica_o_centro_medico", "media",
                     "Razón social sin sufijo legal (\"S.R.L\") + ubicación -- no verificado con búsqueda dedicada en esta sesión."),
    "J-29877616-1": ("Complejo Médico San Lucas, El Paraíso", "clinica_o_centro_medico", "media",
                      "Razón social sin sufijo legal (\"C.A.\") + sector -- no verificado con búsqueda dedicada en esta sesión."),
}

resumen = pd.read_excel(RUTA_ORIGINAL, sheet_name="Resumen")
reviews = pd.read_excel(RUTA_ORIGINAL, sheet_name="Reviews")

resumen["nombre_comercial_candidato"] = resumen["ID"].map(lambda i: CANDIDATOS.get(i, (None,))[0])
resumen["tipo_candidato"] = resumen["ID"].map(lambda i: CANDIDATOS.get(i, (None, None))[1])
resumen["confianza_candidato"] = resumen["ID"].map(lambda i: CANDIDATOS.get(i, (None, None, None))[2])
resumen["nota_candidato"] = resumen["ID"].map(lambda i: CANDIDATOS.get(i, (None, None, None, None))[3])

# Las columnas nuevas solo tienen sentido para encontrado=False -- para el
# resto (True) se dejan vacías explícitamente, no hace falta candidato.
mask_encontrado = resumen["encontrado"]
for col in ["nombre_comercial_candidato", "tipo_candidato", "confianza_candidato", "nota_candidato"]:
    resumen.loc[mask_encontrado, col] = None

n_con_candidato = resumen.loc[~mask_encontrado, "nombre_comercial_candidato"].notna().sum()
n_sin_candidato_claro = (~mask_encontrado).sum() - n_con_candidato
print(f"no_encontrado total: {(~mask_encontrado).sum()}")
print(f"  con candidato para reintentar: {n_con_candidato}")
print(f"  sin candidato claro (dirección residencial/genérica): {n_sin_candidato_claro}")
print()
print(resumen.loc[~mask_encontrado, "confianza_candidato"].value_counts(dropna=False))

with pd.ExcelWriter(RUTA_SALIDA, engine="openpyxl") as writer:
    resumen.to_excel(writer, sheet_name="Resumen", index=False)
    reviews.to_excel(writer, sheet_name="Reviews", index=False)

print(f"\nEscrito: {RUTA_SALIDA}")
