import os
from datetime import date
from typing import List, Literal, Optional

import bcrypt
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from scipy.optimize import Bounds, LinearConstraint, milp
from supabase import Client, create_client

# Pesos del modelo de ranking v2 (medi-match-api/modelo_miguel/Optimizador_Web_Pesos3.ipynb,
# portado 2026-09-13 — reemplaza el índice anterior de 3 variables,
# Programa_optimizador_asignacion_proveedores_1.ipynb, ver docs/DECISIONES.md).
# 4 variables: coste, valoración, capacidad (relativa a la demanda
# pronosticada de las próximas 4 semanas) y la cuota que le toca a cada
# proveedor en un MILP de asignación óptima (scipy) resuelto en cada
# búsqueda. Solo corre cuando pasar_modelo=true (existe forecast, ver
# forecast_demanda) — para el resto sigue calcular_orden_simple.
PESO_COSTE = 0.25
PESO_VALORACION = 0.45
PESO_CAPACIDAD = 0.20
PESO_CUOTA_MILP = 0.10

# El cuaderno de referencia usa 30s de límite para el MILP; es demasiado
# para una petición HTTP síncrona. Los mercados con forecast tienen como
# máximo ~13 proveedores, así que 5s ya es un margen amplio.
MILP_TIME_LIMIT_SEGUNDOS = 5

load_dotenv()

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]

# Límite global de resultados de /recomendar (0 = todos los que califiquen).
# Es una sola variable de entorno para todo el sistema, no algo que el
# frontend pueda pedir por búsqueda — se configura en el .env local o en
# Vercel (Project Settings → Environment Variables del proyecto medi-match-api).
CANTIDAD_PROVEEDORES_MOSTRAR = int(os.environ.get("CANTIDAD_PROVEEDORES_MOSTRAR", "0"))

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

app = FastAPI(title="MediMatch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Municipio(BaseModel):
    id_municipio: str
    municipio: str


class RequestRecomendacion(BaseModel):
    # id_municipio, no el nombre: varios municipios distintos comparten el
    # mismo DESCMUNICIPIO en el dataset (p.ej. dos códigos distintos se llaman
    # "CHACAO") — filtrar por texto los mezclaba y hacía inconsistente
    # pasar_modelo dentro de una misma búsqueda.
    id_municipio: str
    tratamiento: str
    umbral_valoracion: float = 0


class ResponseProveedor(BaseModel):
    id_proveedor: int
    nombre_proveedor: str
    coste_estimado: float
    valoracion: float
    capacidad_restante: int
    ahorro_pct: float  # % de ahorro vs el costo promedio de esta búsqueda (negativo = más caro que el promedio)
    # % de ahorro vs el promedio de TODOS los proveedores del mismo
    # tratamiento en el municipio (no solo los que quedaron en esta
    # búsqueda) -- cálculo aparte, ver _promedio_costo_municipio. None si el
    # municipio no tiene otro proveedor con el mismo tratamiento para comparar.
    ahorro_referencia_municipio_pct: Optional[float] = None
    indice_ranking: float
    modelo_aplicado: bool
    # NULL para los proveedores sin match en el scrape de Google Places (ver
    # docs/DECISIONES.md, 2026-09-13) -- el frontend cae a un link de
    # búsqueda por nombre cuando vienen vacíos.
    google_place_id: Optional[str] = None
    google_maps_url: Optional[str] = None
    # Texto plano, no depende de que el iframe de preview de Maps (sin API
    # key) decida mostrar la ficha del lugar -- resultó no ser confiable
    # (ver docs/DECISIONES.md, 2026-09-13).
    direccion: Optional[str] = None
    telefono: Optional[str] = None


class RequestReserva(BaseModel):
    id_proveedor: int
    municipio: str
    tratamiento: str
    id_cliente: str
    id_gestor: str
    fecha_servicio: date = Field(default_factory=date.today)
    tipo_servicio: Literal["programado", "urgencia"] = "programado"
    observaciones: Optional[str] = None
    id_reserva: Optional[str] = None  # si viene, es una edición, no una reserva nueva


class ResponseReserva(BaseModel):
    status: str
    mensaje: str
    id_reserva: Optional[str] = None


class Cliente(BaseModel):
    id_cliente: str
    poliza: str
    documento: str
    nombre_completo: str
    tipo_usuario: str


class LoginRequest(BaseModel):
    email: str
    password: str


class Gestor(BaseModel):
    id_gestor: str
    nombre: str
    email: str


class AsignacionDetalle(BaseModel):
    id_reserva: str
    id_proveedor: int
    fecha_reserva: str
    fecha_servicio: str
    tipo_servicio: str
    tratamiento: str
    observaciones: Optional[str] = None
    nombre_proveedor: str
    municipio: str
    id_cliente: str
    nombre_cliente: str
    poliza: str
    documento: str
    tipo_usuario: str
    id_gestor: str
    nombre_gestor: str


def _beneficio(valores: np.ndarray, invertir: bool = False) -> np.ndarray:
    # Normalización min-max 0..1 contra el resto de candidatos de esta
    # búsqueda. Más alto = mejor. invertir=True para variables donde menor
    # es mejor (coste): se resta de 1 después de normalizar.
    if valores.max() == valores.min():
        resultado = np.ones_like(valores)
    else:
        resultado = (valores - valores.min()) / (valores.max() - valores.min())
    return 1 - resultado if invertir else resultado


def calcular_ahorro_pct(costos: List[float]) -> List[float]:
    """% de ahorro de cada costo vs el costo promedio de este mismo grupo de
    candidatos (positivo = más barato que el promedio, negativo = más caro).
    Ya no alimenta el ranking (ver calcular_ranking_modelo) — se mantiene
    solo para mostrarlo en la tarjeta de proveedor del frontend."""
    costos_arr = np.array(costos, dtype=float)
    promedio = costos_arr.mean()
    if promedio == 0:
        return [0.0] * len(costos)
    return (((promedio - costos_arr) / promedio) * 100).tolist()


# --- Ahorro de referencia (municipio) -----------------------------------
# Bloque totalmente separado de calcular_ranking_modelo/calcular_ahorro_pct
# de arriba -- no toca el optimizador del compañero ni el índice de ranking.
# Motivo: calcular_ahorro_pct compara contra el promedio de los candidatos
# que quedaron en ESTA búsqueda después de filtrar por valoración/capacidad;
# cuando esa búsqueda no pasa por el modelo (pasar_modelo=false) suele quedar
# 1 solo candidato (ver docs/DECISIONES.md), y comparar un valor contra el
# promedio de sí mismo da 0% siempre -- no es un bug del cálculo, es que no
# hay nada más para comparar en ese grupo chico. Este bloque compara en
# cambio contra el promedio de TODOS los proveedores que ofrecen el mismo
# tratamiento en el mismo municipio (sin el filtro de valoración/capacidad),
# así sigue siendo útil aunque la búsqueda final muestre pocos proveedores.
def _promedio_costo_municipio(id_municipio: str, id_tratamiento: int) -> Optional[float]:
    filas = (
        supabase.table("costo_tratamientos")
        .select("coste_medio")
        .eq("id_municipio", id_municipio)
        .eq("id_tratamiento", id_tratamiento)
        .execute()
        .data
    )
    if len(filas) < 2:
        # Un solo proveedor en todo el municipio -- no hay contra qué
        # comparar, no se inventa un ahorro.
        return None
    return sum(f["coste_medio"] for f in filas) / len(filas)


def calcular_ahorro_referencia_pct(costo: float, promedio_municipio: Optional[float]) -> Optional[float]:
    if promedio_municipio is None or promedio_municipio == 0:
        return None
    return ((promedio_municipio - costo) / promedio_municipio) * 100


def _resolver_milp_cuota(
    costos: np.ndarray,
    valoraciones: np.ndarray,
    capacidad_4_semanas: np.ndarray,
    demanda_operativa: int,
    umbral_valoracion: float,
) -> np.ndarray:
    """Corre el MILP de asignación óptima (minimiza coste total repartiendo
    la demanda pronosticada entre proveedores, respetando su capacidad y una
    valoración media mínima) y devuelve cuántos servicios le tocó a cada
    proveedor en esa asignación (0 si ninguno). Sin demanda operativa o con
    el problema infactible, devuelve todo ceros: el score de cuota queda
    neutro para todos, no rompe el resto del índice."""
    n = len(costos)
    if n == 0 or demanda_operativa <= 0:
        return np.zeros(n)

    capacidades_enteras = np.floor(capacidad_4_semanas + 1e-9)
    restricciones = LinearConstraint(
        np.vstack([np.ones(n), valoraciones]),
        [demanda_operativa, demanda_operativa * umbral_valoracion],
        [demanda_operativa, np.inf],
    )
    resultado = milp(
        c=costos,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), capacidades_enteras),
        constraints=restricciones,
        options={"time_limit": MILP_TIME_LIMIT_SEGUNDOS},
    )
    if not resultado.success:
        return np.zeros(n)
    return np.rint(resultado.x)


def calcular_ranking_modelo(
    costos: List[float],
    valoraciones: List[float],
    capacidad_restante: List[int],
    demanda_forecast: float,
    demanda_operativa: int,
    umbral_valoracion: float,
) -> List[float]:
    """Modelo de un compañero (modelo_miguel/Optimizador_Web_Pesos3.ipynb):
    coste 25% + valoración 45% + capacidad (relativa a la demanda pronosticada
    de las próximas 4 semanas) 20% + cuota de un MILP de asignación óptima
    10%, cada score normalizado min-max contra el resto de candidatos de esta
    búsqueda. Reemplaza el índice anterior (valoración 50 / ahorro 30 /
    capacidad 20) — ver docs/DECISIONES.md, 2026-09-13.

    Usa capacidad_restante (lo que de verdad queda), no la capacidad
    contratada completa: esta última nunca se resetea en producción (ver
    docs/DECISIONES.md), así que usarla tal cual recomendaría proveedores ya
    agotados como si tuvieran toda su capacidad libre."""
    costos_arr = np.array(costos, dtype=float)
    valoraciones_arr = np.array(valoraciones, dtype=float)
    capacidad_4_semanas = np.array(capacidad_restante, dtype=float) * 4 / 52

    score_coste = _beneficio(costos_arr, invertir=True)
    score_valoracion = _beneficio(valoraciones_arr)

    capacidad_relativa = (
        capacidad_4_semanas / demanda_forecast
        if demanda_forecast > 0
        else np.zeros_like(capacidad_4_semanas)
    )
    score_capacidad = _beneficio(capacidad_relativa)

    cuota = _resolver_milp_cuota(
        costos_arr, valoraciones_arr, capacidad_4_semanas, demanda_operativa, umbral_valoracion
    )
    score_cuota = cuota / cuota.max() if cuota.max() > 0 else np.zeros_like(cuota)

    return (
        PESO_COSTE * score_coste
        + PESO_VALORACION * score_valoracion
        + PESO_CAPACIDAD * score_capacidad
        + PESO_CUOTA_MILP * score_cuota
    ).tolist()


def calcular_orden_simple(valoraciones: List[float], costos: List[float]) -> List[float]:
    """Para combinaciones con pasar_modelo=false: no corren el modelo (pocas
    alternativas, no vale la pena). Orden transparente por valoración y costo,
    sin pretender ser un score del modelo — solo para poder ordenar la lista."""
    orden = sorted(range(len(valoraciones)), key=lambda i: (-valoraciones[i], costos[i]))
    posicion = {i: pos for pos, i in enumerate(orden)}
    n = len(valoraciones)
    return [1 - posicion[i] / n for i in range(n)]


class ConfigDiagnostico(BaseModel):
    cantidad_proveedores_mostrar: int
    vercel_env: Optional[str] = None
    vercel_region: Optional[str] = None
    vercel_git_commit_sha: Optional[str] = None
    vercel_git_commit_ref: Optional[str] = None


@app.get("/config", response_model=ConfigDiagnostico)
def config_diagnostico() -> ConfigDiagnostico:
    # Diagnóstico en vivo, no un secreto: sirve para confirmar sin
    # adivinar, desde curl, qué env vars y qué commit está sirviendo
    # realmente la instancia que responde (útil porque Vercel puede tardar
    # en propagar un deploy nuevo a todas las instancias — ver
    # docs/DECISIONES.md, 2026-09-07/08).
    return ConfigDiagnostico(
        cantidad_proveedores_mostrar=CANTIDAD_PROVEEDORES_MOSTRAR,
        vercel_env=os.environ.get("VERCEL_ENV"),
        vercel_region=os.environ.get("VERCEL_REGION"),
        vercel_git_commit_sha=os.environ.get("VERCEL_GIT_COMMIT_SHA"),
        vercel_git_commit_ref=os.environ.get("VERCEL_GIT_COMMIT_REF"),
    )


@app.get("/municipios", response_model=List[Municipio])
def listar_municipios() -> List[Municipio]:
    filas = supabase.table("proveedores").select("id_municipio, municipio").execute().data
    # (id_municipio, municipio): el nombre solo no alcanza como identidad —
    # ver comentario en RequestRecomendacion.
    vistos = {(f["id_municipio"], f["municipio"]) for f in filas}
    return sorted(
        (Municipio(id_municipio=id_municipio, municipio=nombre) for id_municipio, nombre in vistos),
        key=lambda m: (m.municipio, m.id_municipio),
    )


@app.get("/tratamientos", response_model=List[str])
def listar_tratamientos(id_municipio: str = Query(...)) -> List[str]:
    ids_proveedor = [
        p["id_proveedor"]
        for p in supabase.table("proveedores")
        .select("id_proveedor")
        .eq("id_municipio", id_municipio)
        .execute()
        .data
    ]
    if not ids_proveedor:
        return []

    filas = (
        supabase.table("costo_tratamientos")
        .select("tratamiento")
        .in_("id_proveedor", ids_proveedor)
        .execute()
        .data
    )
    return sorted({f["tratamiento"] for f in filas})


@app.post("/login", response_model=Gestor)
def login(request: LoginRequest) -> Gestor:
    filas = (
        supabase.table("gestores")
        .select("id_gestor, nombre, email, password_hash")
        .eq("email", request.email)
        .execute()
        .data
    )
    if not filas or not bcrypt.checkpw(
        request.password.encode("utf-8"), filas[0]["password_hash"].encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Credenciales inválidas")

    gestor = filas[0]
    return Gestor(id_gestor=gestor["id_gestor"], nombre=gestor["nombre"], email=gestor["email"])


@app.get("/clientes", response_model=List[Cliente])
def buscar_clientes(q: str = Query(..., min_length=2)) -> List[Cliente]:
    # Autocompleta el formulario: un solo cuadro de búsqueda que matchea por
    # póliza, documento o nombre (lo que sea que el usuario haya tipeado).
    filas = (
        supabase.table("clientes")
        .select("id_cliente, poliza, documento, nombre_completo, tipo_usuario")
        .or_(f"poliza.ilike.%{q}%,documento.ilike.%{q}%,nombre_completo.ilike.%{q}%")
        .limit(10)
        .execute()
        .data
    )
    return filas


_CAMPOS_PROVEEDOR_BASE = "id_proveedor, nombre_proveedor, valoracion"
_CAMPOS_PROVEEDOR_GOOGLE = ["google_place_id", "google_maps_url"]  # migración 022
_CAMPOS_PROVEEDOR_DIRECCION = ["direccion", "telefono"]  # migración 023
_TODOS_LOS_CAMPOS_OPCIONALES = _CAMPOS_PROVEEDOR_GOOGLE + _CAMPOS_PROVEEDOR_DIRECCION


def _seleccionar_proveedores(id_municipio: str, umbral_valoracion: float) -> List[dict]:
    # Cubre el período entre desplegar este código y aplicar las migraciones
    # 022/023 a mano (medi-match-db/migrations) — sin esas columnas todavía,
    # PostgREST devuelve error, no filas sin ellas. Se intenta con todo y se
    # va cayendo a menos columnas (022 sin 023, o ninguna) en vez de romper
    # /recomendar por completo.
    intentos = [
        _TODOS_LOS_CAMPOS_OPCIONALES,
        _CAMPOS_PROVEEDOR_GOOGLE,
        [],
    ]
    for campos_extra in intentos:
        try:
            filas = (
                supabase.table("proveedores")
                .select(", ".join([_CAMPOS_PROVEEDOR_BASE, *campos_extra]))
                .eq("id_municipio", id_municipio)
                .gte("valoracion", umbral_valoracion)
                .execute()
                .data
            )
        except Exception:
            continue
        for fila in filas:
            for campo in _TODOS_LOS_CAMPOS_OPCIONALES:
                fila.setdefault(campo, None)
        return filas
    return []


@app.post("/recomendar", response_model=List[ResponseProveedor])
def recomendar(request: RequestRecomendacion) -> List[ResponseProveedor]:
    proveedores = _seleccionar_proveedores(request.id_municipio, request.umbral_valoracion)
    if not proveedores:
        return []

    ids_proveedor = [p["id_proveedor"] for p in proveedores]
    catalogo = (
        supabase.table("costo_tratamientos")
        .select("id_proveedor, id_tratamiento, coste_medio, pasar_modelo")
        .in_("id_proveedor", ids_proveedor)
        .eq("tratamiento", request.tratamiento)
        .execute()
        .data
    )
    if not catalogo:
        return []

    proveedores_por_id = {p["id_proveedor"]: p for p in proveedores}
    # Si un proveedor ofrece el tratamiento buscado a más de un costo (varias
    # líneas del dataset agrupadas distinto), nos quedamos con la más barata.
    catalogo_por_proveedor = {}
    for fila in catalogo:
        actual = catalogo_por_proveedor.get(fila["id_proveedor"])
        if actual is None or fila["coste_medio"] < actual["coste_medio"]:
            catalogo_por_proveedor[fila["id_proveedor"]] = fila

    # La capacidad es por par (proveedor, tratamiento), no un pool compartido
    # por todo lo que ofrece el proveedor.
    ids_tratamiento = list({fila["id_tratamiento"] for fila in catalogo_por_proveedor.values()})
    capacidades_filas = (
        supabase.table("capacidades")
        .select("id_proveedor, id_tratamiento, capacidad_maxima, capacidad_restante")
        .in_("id_proveedor", list(catalogo_por_proveedor.keys()))
        .in_("id_tratamiento", ids_tratamiento)
        .execute()
        .data
    )
    capacidad_por_par = {(c["id_proveedor"], c["id_tratamiento"]): c for c in capacidades_filas}

    combinados = [
        (proveedores_por_id[id_proveedor], fila)
        for id_proveedor, fila in catalogo_por_proveedor.items()
    ]

    def capacidad_de(p, fila):
        return capacidad_por_par.get((p["id_proveedor"], fila["id_tratamiento"]))

    # pasar_modelo es una propiedad de la combinación municipio+tratamiento
    # buscada: todas las filas del catálogo que la componen comparten el
    # mismo valor (verificado contra la base real), así que basta con leerlo
    # de la primera — antes de filtrar nada.
    modelo_aplicado = bool(combinados[0][1]["pasar_modelo"])

    if modelo_aplicado:
        # El modelo v2 no tiene sentido para un proveedor ya agotado —se
        # filtra antes de rankear. El camino sin modelo no cambia de
        # comportamiento (no se pidió tocarlo).
        combinados = [
            (p, fila)
            for p, fila in combinados
            if (capacidad_de(p, fila) or {}).get("capacidad_restante", 0) > 0
        ]
        if not combinados:
            return []

    capacidad_restante = [
        (capacidad_de(p, fila) or {}).get("capacidad_restante", 0) for p, fila in combinados
    ]
    costos = [fila["coste_medio"] for _, fila in combinados]
    valoraciones = [p["valoracion"] for p, _ in combinados]
    ahorros_pct = calcular_ahorro_pct(costos)

    # id_tratamiento es una propiedad del catálogo (no del modelo) — se lee
    # una sola vez y se usa tanto para el forecast (solo si modelo_aplicado)
    # como para el ahorro de referencia de abajo (siempre).
    id_tratamiento = combinados[0][1]["id_tratamiento"]
    promedio_municipio = _promedio_costo_municipio(request.id_municipio, id_tratamiento)
    ahorros_referencia_pct = [calcular_ahorro_referencia_pct(c, promedio_municipio) for c in costos]

    if modelo_aplicado:
        try:
            forecast_filas = (
                supabase.table("forecast_demanda")
                .select("demanda_forecast, demanda_operativa")
                .eq("id_municipio", request.id_municipio)
                .eq("id_tratamiento", id_tratamiento)
                .execute()
                .data
            )
        except Exception:
            # Cubre el período entre desplegar este código y aplicar la
            # migración 021 a mano (medi-match-db/migrations) — sin la
            # tabla todavía, PostgREST devuelve error, no una lista vacía.
            forecast_filas = []
        if forecast_filas:
            indices = calcular_ranking_modelo(
                costos,
                valoraciones,
                capacidad_restante,
                demanda_forecast=forecast_filas[0]["demanda_forecast"],
                demanda_operativa=forecast_filas[0]["demanda_operativa"],
                umbral_valoracion=request.umbral_valoracion,
            )
        else:
            # No debería pasar — pasar_modelo=true implica forecast cargado
            # (verificado 1:1, ver docs/DECISIONES.md) — pero si pasa, no
            # romper la búsqueda por un dato faltante.
            indices = calcular_orden_simple(valoraciones, costos)
    else:
        indices = calcular_orden_simple(valoraciones, costos)

    respuesta = [
        ResponseProveedor(
            id_proveedor=p["id_proveedor"],
            nombre_proveedor=p["nombre_proveedor"],
            coste_estimado=fila["coste_medio"],
            valoracion=p["valoracion"],
            capacidad_restante=restante,
            ahorro_pct=ahorro,
            ahorro_referencia_municipio_pct=ahorro_referencia,
            indice_ranking=indice,
            modelo_aplicado=modelo_aplicado,
            google_place_id=p.get("google_place_id"),
            google_maps_url=p.get("google_maps_url"),
            direccion=p.get("direccion"),
            telefono=p.get("telefono"),
        )
        for (p, fila), restante, ahorro, ahorro_referencia, indice in zip(
            combinados, capacidad_restante, ahorros_pct, ahorros_referencia_pct, indices
        )
    ]
    ordenado = sorted(respuesta, key=lambda p: p.indice_ranking, reverse=True)
    if CANTIDAD_PROVEEDORES_MOSTRAR > 0:
        ordenado = ordenado[:CANTIDAD_PROVEEDORES_MOSTRAR]
    return ordenado


@app.post("/reservar", response_model=ResponseReserva)
def reservar(request: RequestReserva) -> ResponseReserva:
    # asignaciones necesita el código del tratamiento (capacidades está keyed
    # por id_tratamiento, no por el texto) — se resuelve a partir de
    # (id_proveedor, tratamiento) contra el catálogo, sin pedirle ese código
    # al frontend.
    fila_tratamiento = (
        supabase.table("costo_tratamientos")
        .select("id_tratamiento, id_municipio")
        .eq("id_proveedor", request.id_proveedor)
        .eq("tratamiento", request.tratamiento)
        .limit(1)
        .execute()
        .data
    )
    if not fila_tratamiento:
        return ResponseReserva(status="error", mensaje="Ese proveedor no ofrece ese tratamiento")
    id_tratamiento = fila_tratamiento[0]["id_tratamiento"]
    id_municipio = fila_tratamiento[0]["id_municipio"]

    es_edicion = request.id_reserva is not None
    par_anterior = None

    if es_edicion:
        actual = (
            supabase.table("asignaciones")
            .select("id_proveedor, id_tratamiento")
            .eq("id_reserva", request.id_reserva)
            .eq("estado", "activo")
            .execute()
            .data
        )
        if not actual:
            return ResponseReserva(status="error", mensaje="La asignación no existe")
        par_anterior = (actual[0]["id_proveedor"], actual[0]["id_tratamiento"])

    # Si es una edición y el par (proveedor, tratamiento) no cambió, no hace
    # falta chequear cupo (no se está consumiendo capacidad nueva). El
    # trigger de UPDATE en medi-match-db se encarga de mover la capacidad
    # entre pares si sí cambió; el de INSERT, de descontarla en una reserva
    # nueva.
    cambia_par = not es_edicion or par_anterior != (request.id_proveedor, id_tratamiento)
    if cambia_par:
        capacidad = (
            supabase.table("capacidades")
            .select("capacidad_restante")
            .eq("id_proveedor", request.id_proveedor)
            .eq("id_tratamiento", id_tratamiento)
            .execute()
        )
        if not capacidad.data or capacidad.data[0]["capacidad_restante"] <= 0:
            return ResponseReserva(status="error", mensaje="Sin cupos disponibles")

    payload = {
        "id_cliente": request.id_cliente,
        "id_proveedor": request.id_proveedor,
        "id_tratamiento": id_tratamiento,
        "id_municipio": id_municipio,
        "tratamiento": request.tratamiento,
        "id_gestor": request.id_gestor,
        "fecha_servicio": request.fecha_servicio.isoformat(),
        "tipo_servicio": request.tipo_servicio,
        "observaciones": request.observaciones,
    }

    if es_edicion:
        supabase.table("asignaciones").update(payload).eq("id_reserva", request.id_reserva).execute()
        return ResponseReserva(
            status="success", mensaje="Asignación actualizada", id_reserva=request.id_reserva
        )

    fila = supabase.table("asignaciones").insert(payload).execute().data[0]
    return ResponseReserva(status="success", mensaje="Asignado", id_reserva=fila["id_reserva"])


@app.get("/asignaciones", response_model=List[AsignacionDetalle])
def listar_asignaciones(id_gestor: Optional[str] = None) -> List[AsignacionDetalle]:
    query = (
        supabase.table("asignaciones")
        .select(
            "id_reserva, id_proveedor, fecha_reserva, fecha_servicio, tipo_servicio,"
            " tratamiento, observaciones, id_cliente, id_gestor,"
            " proveedores!asignaciones_id_proveedor_fkey(nombre_proveedor, municipio),"
            " clientes(nombre_completo, poliza, documento, tipo_usuario),"
            " gestores(nombre)"
        )
        .eq("estado", "activo")
        .order("fecha_reserva", desc=True)
    )
    if id_gestor:
        query = query.eq("id_gestor", id_gestor)

    filas = query.execute().data
    return [
        AsignacionDetalle(
            id_reserva=f["id_reserva"],
            id_proveedor=f["id_proveedor"],
            fecha_reserva=f["fecha_reserva"],
            fecha_servicio=f["fecha_servicio"],
            tipo_servicio=f["tipo_servicio"],
            tratamiento=f["tratamiento"],
            observaciones=f["observaciones"],
            nombre_proveedor=f["proveedores"]["nombre_proveedor"],
            municipio=f["proveedores"]["municipio"],
            id_cliente=f["id_cliente"],
            nombre_cliente=f["clientes"]["nombre_completo"],
            poliza=f["clientes"]["poliza"],
            documento=f["clientes"]["documento"],
            tipo_usuario=f["clientes"]["tipo_usuario"],
            id_gestor=f["id_gestor"],
            nombre_gestor=f["gestores"]["nombre"],
        )
        for f in filas
    ]


@app.delete("/asignaciones/{id_reserva}", response_model=ResponseReserva)
def eliminar_asignacion(id_reserva: str) -> ResponseReserva:
    # Soft delete: no se borra la fila (queda como auditoría), solo se marca
    # estado='eliminado'. El trigger de UPDATE en medi-match-db libera la
    # capacidad automáticamente en esa transición.
    supabase.table("asignaciones").update({"estado": "eliminado"}).eq("id_reserva", id_reserva).execute()
    return ResponseReserva(status="success", mensaje="Asignación eliminada")
