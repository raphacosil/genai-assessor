import os
from dotenv import load_dotenv
import psycopg2
from typing import Optional
from langchain.tools import tool
from pydantic import BaseModel, Field

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")  

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def close_conn(conn):
    try:
        if conn:
            conn.close()
    except Exception:
        pass

class AddTransactionArgs(BaseModel):
    amount: float = Field(..., description="Valor da transação (use positivo).")
    source_text: str = Field(..., description="Texto original do usuário.")
    occurred_at: Optional[str] = Field(
        default=None,
        description="Timestamp ISO 8601; se ausente, usa NOW() no banco."
    )
    type_id: Optional[int] = Field(default=None, description="ID em transaction_types (1=INCOME, 2=EXPENSES, 3=TRANSFER).")
    type_name: Optional[str] = Field(default=None, description="Nome do tipo: INCOME | EXPENSES | TRANSFER.")
    category_id: Optional[int] = Field(default=None, description="FK de categories (opcional).")
    description: Optional[str] = Field(default=None, description="Descrição (opcional).")
    payment_method: Optional[str] = Field(default=None, description="Forma de pagamento (opcional).")


class QueryTransactionsArgs(BaseModel):
    text: Optional[str] = Field(default=None, description="Texto a buscar em source_text ou description (opcional).")
    type_name: Optional[str] = Field(default=None, description="Nome do tipo: INCOME | EXPENSES | TRANSFER (opcional).")
    date_local: Optional[str] = Field(default=None, description="Data local YYYY-MM-DD (America/Sao_Paulo) (opcional).")
    date_from_local: Optional[str] = Field(default=None, description="Data inicial local YYYY-MM-DD (America/Sao_Paulo) (opcional).")
    date_to_local: Optional[str] = Field(default=None, description="Data final local YYYY-MM-DD (America/Sao_Paulo) (opcional).")
    limit: int = Field(default=20, description="Número máximo de registros a retornar.")

def _resolve_type_id(cur, type_id: Optional[int], type_name: Optional[str]) -> Optional[int]:
    if type_name:
        t = type_name.strip().upper()
        if t == "EXPENSE":
            t = "EXPENSES"
        cur.execute("SELECT id FROM transaction_types WHERE UPPER(type)=%s LIMIT 1;", (t,))
        row = cur.fetchone()
        return row[0] if row else None
    if type_id:
        return int(type_id)
    return 2


@tool("add_transaction", args_schema=AddTransactionArgs)
def add_transaction(
    amount: float,
    source_text: str,
    occurred_at: Optional[str] = None,
    type_id: Optional[int] = None,
    type_name: Optional[str] = None,
    category_id: Optional[int] = None,
    description: Optional[str] = None,
    payment_method: Optional[str] = None,
) -> dict:
    """
    Adiciona uma transação (amount positivo) com os dados fornecidos.
    """
    conn = get_conn()
    cur = conn.cursor()
    try:
        resolved_type_id = _resolve_type_id(cur, type_id, type_name)
        if not resolved_type_id:
            return {"status": "error", "message": "Tipo inválido (use type_id ou type_name: INCOME/EXPENSES/TRANSFER)."}

        if occurred_at:
            cur.execute(
                """
                INSERT INTO transactions
                    (amount, type, category_id, description, payment_method, occurred_at, source_text)
                VALUES
                    (%s, %s, %s, %s, %s, %s::timestamptz, %s)
                RETURNING id, occurred_at;
                """,
                (amount, resolved_type_id, category_id, description, payment_method, occurred_at, source_text),
            )
        else:
            cur.execute(
                """
                INSERT INTO transactions
                    (amount, type, category_id, description, payment_method, occurred_at, source_text)
                VALUES
                    (%s, %s, %s, %s, %s, NOW(), %s)
                RETURNING id, occurred_at;
                """,
                (amount, resolved_type_id, category_id, description, payment_method, source_text),
            )

        new_id, occurred = cur.fetchone()
        conn.commit()
        return {"status": "ok", "id": new_id, "occurred_at": str(occurred)}

    except Exception as e:
        conn.rollback()
        return {"status": "error", "message": str(e)}
    finally:
        try:
            cur.close()
            close_conn(conn)
        except Exception:
            pass

@tool("query_transactions", args_schema=QueryTransactionsArgs)
def query_transactions(
    text: Optional[str] = None,
    type_name: Optional[str] = None,
    date_local: Optional[str] = None,
    date_from_local: Optional[str] = None,
    date_to_local: Optional[str] = None,
    limit: int = 20
) -> dict:
    """Consulta as transações com filtros por texto (source_text/description), tipo e datas locais (America/Sao_Paulo).
    Os dados devem vir na seguinte ordem:
     - Intervalo (date_from_local/date_to_local): ASC (cronológico).
     - Caso contrário: DESC (mais recente primeiro)"""
     
    conn = get_conn()
    cur = conn.cursor()

    try:
        base_query = """
        SELECT * FROM transactions t
        JOIN transaction_types tt ON tt.id = t.type
        WHERE 1=1
        """
        conditions, params = [], []

        if text:
            conditions.append("(t.source_text ILIKE %s OR t.description ILIKE %s)")
            params.extend([f"%{text}%", f"%{text}%"])

        if type_name:
            conditions.append("tt.type ILIKE %s")
            params.append(f"%{type_name}%")

        if date_local:
            conditions.append("t.occurred_at::date = (%s::date AT TIME ZONE 'America/Sao_Paulo')")
            params.append(date_local)

        if date_from_local and date_to_local:
            conditions.append("""t.occurred_at::date BETWEEN
                                (%s::date AT TIME ZONE 'America/Sao_Paulo')
                                AND (%s::date AT TIME ZONE 'America/Sao_Paulo')""")
            params.extend([date_from_local, date_to_local])
            order_clause = "ORDER BY t.occurred_at ASC"
        else:
            order_clause = "ORDER BY t.occurred_at DESC"

        query = base_query + " AND ".join(conditions) + f" {order_clause} LIMIT %s"
        params.append(limit)

        cur.execute(query, params)
        rows = cur.fetchall()

        return {"transactions": [
            {
                "id": r[0],
                "amount": float(r[1]),
                "type_name": r[2],
                "category_id": r[3],
                "description": r[4],
                "payment_method": r[5],
                "occurred_at": r[6].isoformat(),
                "source_text": r[7]
            }
            for r in rows
        ]}

    except Exception as e:
        return {"status": "error", "message": str(e)}

    finally:
        cur.close()
        close_conn(conn)
        

@tool("total_balance")
def total_balance() -> dict:
    """
    Retorna o saldo total (INCOME - EXPENSES) em todo o histórico (ignora TRANSFER).
    """
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT 
                COALESCE(SUM(CASE WHEN tt.type = 'INCOME' THEN t.amount END), 0)
                - COALESCE(SUM(CASE WHEN tt.type = 'EXPENSES' THEN t.amount END), 0) AS balance
            FROM transactions t
            JOIN transaction_types tt ON tt.id = t.type
            """)
        return {"saldo_total": float(cur.fetchone()[0])}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cur.close()
        close_conn(conn)

@tool("daily_balance")
def daily_balance(date_local: str) -> dict:
    """
    Retorna o saldo (INCOME - EXPENSES) do dia local informado (YYYY-MM-DD) em America_Sao_Paulo.
    Ignora TRANSFER (type=3).
    """
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
        SELECT 
            COALESCE(SUM(CASE WHEN tt.type = 'INCOME' THEN t.amount END), 0)
            - COALESCE(SUM(CASE WHEN tt.type = 'EXPENSES' THEN t.amount END), 0) AS balance
        FROM transactions t
        JOIN transaction_types tt ON tt.id = t.type
        WHERE t.occurred_at::date = %s::date
        """, (date_local,))
        return {"saldo_dia": float(cur.fetchone()[0]), "date": date_local}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cur.close()
        close_conn(conn)

@tool("in_time_interval_balance")
def in_time_interval_balance(date_from_local: str, date_to_local: str) -> dict:
    """
    Retorna o saldo (INCOME - EXPENSES) do intervalo de datas local informado (YYYY-MM-DD) em America_Sao_Paulo.
    Ignora TRANSFER (type=3).
    """
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
        SELECT 
            COALESCE(SUM(CASE WHEN tt.type = 'INCOME' THEN t.amount END), 0)
            - COALESCE(SUM(CASE WHEN tt.type = 'EXPENSES' THEN t.amount END), 0) AS balance
        FROM transactions t
        JOIN transaction_types tt ON tt.id = t.type
        WHERE t.occurred_at::date BETWEEN %s::date AND %s::date
        """, (date_from_local, date_to_local))
        return {"saldo_intervalo": float(cur.fetchone()[0]), "date_from": date_from_local, "date_to": date_to_local}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cur.close()
        close_conn(conn)

@tool("in_time_interval_income")
def in_time_interval_income(date_from_local: str, date_to_local: str) -> dict:
    """
    Retorna o total de INCOME do intervalo de datas local informado (YYYY-MM-DD) em America_Sao_Paulo.
    """
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
        SELECT 
            COALESCE(SUM(CASE WHEN tt.type = 'INCOME' THEN t.amount END), 0)
        FROM transactions t
        JOIN transaction_types tt ON tt.id = t.type
        WHERE t.occurred_at::date BETWEEN %s::date AND %s::date
        """, (date_from_local, date_to_local))
        return {"total_income": float(cur.fetchone()[0]), "date_from": date_from_local, "date_to": date_to_local}

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cur.close()
        close_conn(conn)

@tool("in_time_interval_expenses")
def in_time_interval_expenses(date_from_local: str, date_to_local: str) -> dict:
    """
    Retorna o total de EXPENSES do intervalo de datas local informado (YYYY-MM-DD) em America_Sao_Paulo.
    """
    conn = get_conn()
    cur = conn.cursor()
    try:
        cur.execute("""
        SELECT 
            COALESCE(SUM(CASE WHEN tt.type = 'EXPENSES' THEN t.amount END), 0)
        FROM transactions t
        JOIN transaction_types tt ON tt.id = t.type
        WHERE t.occurred_at::date BETWEEN %s::date AND %s::date
        """, (date_from_local, date_to_local))
        return {"total_expenses": float(cur.fetchone()[0]), "date_from": date_from_local, "date_to": date_to_local}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cur.close()
        close_conn(conn)

TOOLS = [
    add_transaction,
    query_transactions,
    total_balance,
    daily_balance,
    in_time_interval_balance,
    in_time_interval_income,
    in_time_interval_expenses
]
