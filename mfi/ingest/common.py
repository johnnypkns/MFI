import io

from ..db import connect


def madrid_cusec(district: int, section: int) -> str:
    """INE census-section code (CUSEC): province 28 + municipality 079 + district DD + section SSS."""
    return f"28079{int(district):02d}{int(section):03d}"


def already_loaded(table: str, file_id: int) -> bool:
    with connect() as conn:
        return conn.execute(f"SELECT EXISTS (SELECT 1 FROM {table} WHERE file_id=%s)", (file_id,)).fetchone()[0]


def copy_df(table: str, df):
    """Bulk-load a DataFrame with COPY. Column names must match the table. NaN/None -> NULL."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N")  # CSV quoting survives commas/quotes/newlines in names
    buf.seek(0)
    cols = ", ".join(df.columns)
    with connect() as conn:
        with conn.cursor().copy(f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '\\N')") as cp:
            while chunk := buf.read(1 << 20):
                cp.write(chunk)
        conn.commit()
