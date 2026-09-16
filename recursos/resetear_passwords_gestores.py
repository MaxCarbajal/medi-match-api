# Verifica que todos los gestores tengan la contraseña MediMatch2026! y
# resetea los que no la tengan -- pedido explícito del usuario (2026-09-16):
# intentó entrar con su cuenta de gestor y no pudo porque su password_hash no
# correspondía a esa contraseña.
#
# Verificado antes de escribir nada: 4/5 gestores YA tenían MediMatch2026!
# (con hashes distintos entre sí -- cada bcrypt.hashpw usa un salt nuevo, no
# es que compartan el mismo hash). Solo 1 no coincidía:
# maxcarbajalh@gmail.com. Este script solo escribe sobre las cuentas que NO
# coinciden -- no re-hashea las que ya están bien, para minimizar el cambio.
#
# gestores son personas reales con contraseña propia (ver CLAUDE.md) -- esto
# resetea su contraseña real a una conocida, con permiso explícito del
# usuario. No es una cuenta "genérica" que se inventa: se resetea la cuenta
# que ya existe.
#
# Usa la service_role key de medi-match-api/.env (no el conector MCP de
# Supabase de esta sesión, que no ve este proyecto -- ver CLAUDE.md).
import os
import sys

import bcrypt
from dotenv import load_dotenv
from supabase import create_client

PASSWORD_OBJETIVO = "MediMatch2026!"

load_dotenv("../../medi-match-api/.env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

print("Descargando gestores...")
gestores = supabase.table("gestores").select("id_gestor, nombre, email, password_hash").execute().data
print(f"  total: {len(gestores)}")

a_resetear = []
for g in gestores:
    coincide = bcrypt.checkpw(PASSWORD_OBJETIVO.encode("utf-8"), g["password_hash"].encode("utf-8"))
    print(f"  {g['email']}: {'OK, ya tiene ' + PASSWORD_OBJETIVO if coincide else 'NO coincide -- se resetea'}")
    if not coincide:
        a_resetear.append(g)

print(f"\nCuentas a resetear: {len(a_resetear)}/{len(gestores)}")
if not a_resetear:
    print("Nada que hacer -- todos ya tienen la contraseña objetivo.")
    sys.exit(0)

for g in a_resetear:
    print(f"  - {g['nombre']} <{g['email']}>")

confirmar = input(f"\n¿Resetear la contraseña de estas {len(a_resetear)} cuenta(s) a '{PASSWORD_OBJETIVO}'? (escribir 'si' para continuar): ")
if confirmar.strip().lower() != "si":
    print("Cancelado -- no se cambió nada.")
    sys.exit(0)

for g in a_resetear:
    nuevo_hash = bcrypt.hashpw(PASSWORD_OBJETIVO.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    supabase.table("gestores").update({"password_hash": nuevo_hash}).eq("id_gestor", g["id_gestor"]).execute()
    print(f"  actualizado: {g['email']}")

print("\nListo.")
