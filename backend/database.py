import sqlite3
import os
from config import DB_PATH


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS registered_vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE NOT NULL,
            owner_name TEXT,
            vehicle_type TEXT,
            vehicle_model TEXT,
            color TEXT,
            registered_at TEXT DEFAULT CURRENT_TIMESTAMP,
            is_active INTEGER DEFAULT 1
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT,
            status TEXT,
            confidence REAL,
            image_path TEXT,
            accessed_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)

    plates = [
        "KA03MW6574","KA05KT8164","KL34H7617","KA04KL5743",
        "KA51JH9335","KA03AH0214","KA34R3118","KA02HL2744","TN15ME5783",
        "KA51HW9141","TN37CY1558","KL70E8691","KA04KZ8371","KA64L7593",
        "KA01JE1424","KA41U9558","KA05HL6826","KA05QG3646","TN47BC2007",
        "KA01KK9051","KA17HG7625","KL11BU5577","KA03JF7475","KL08S2659",
        "KL13AP1443","KA03AF8376","KL63K0789","25BH3997N","TN38CR2419",
        "KA41EX3916","KA01JN4968","KA01KB1938","KA05JK2381","KL40P2577",
        "KA41HA6973","KA41EL1775","TN65V1921","KA05JV3473","KA41HB9226",
        "KA05LT3154","TN18AA3952","KA41EM6758","KL01CF2526","KL48R6255",
        "KA05KF6019","TN22DJ4554","KA41HA9236","KA01KJ9866","TN39BX3012",
        "KA35J2514","TN13Y6388","KA41EK4405","KA41EZ1790","KA41EG5503",
        "TN22BB7221","KA41EE6176","KA04KL0442","KL73D5111","KL04AW0350",
        "KA41EY8400","KL66A5765","KA41ES5210","KA02LB1869","KA05KC1594",
        "KA51HU2631","KA41EX0974","KA41ER2094","KA41EE8477","KA05LA8287",
        "TN67BL3375","KA41EJ2008","KL25P7644","KL38F6591","KA51HR4876",
        "KA41EV0405","KA41HB0048","KA41EW6446","KA03HK8917","TN83P8005",
        "KA51HH9227","KA05QA8062","AP07CQ5168","KA05KC6793","KA41EU1547",
        "KA41EZ8674","KA41EY4714","KA01ET4825","TN58BE4582","KA41ES1661",
        "KA41ET5465","KA41EM0606","KA16EQ1939","TN23CP7823","KL03AG1146",
        "TN28AS8779","TN31BZ2049","TN45BL6732","KA01JD9831","TN57BQ6003",
        "KA51EL4089","KA05JB0857","KA41EZ8820","KA01JP1618","KA04HN3524",
        "KA01HW6314","KL07BA6379","KA41EW7558","KA05LP7063","TN38DM5011",
        "KL60M2851","KA11K9566","KL53H8727","KA51JD4709","KA41EG8771",
        "KA51EG6983","KA42W6652","KA51EQ2203","TN70D7880","KA01JY0434",
        "KA53JC2007","KA02LD7022","KA03JE7950","KL450335","KA02KY4521",
        "KL08BX6924","KA41ET5292","KA05KZ6834","KA41MG0289","TN69BL4399",
        "KL07CX2106","KL57A4477","KA04NA3663","KA04MZ2067","KA01MN5370",
        "KL47C828","KA04NC7807","KA03NN4141","KA41D9766","KL72D3000",
        "KA05NA6217","KA03NA5037","KA01Z0543","KA05MP4720","KA03NH5104",
        "KA41MG0907","KA02MW7915","TN75D2851","KA02MX4171","KA41ME6438",
        "KA41MF6956","KA41MA5929","KL27B9861","KL08AN2416","KA53C1047",
        "KL07BX9606","KA51MH337","KA05AN0249","KA42EH6122","KA40EC2807",
        "KA05JP1257","TN38DE4757","UP32NS3993","TN05CA3443","KA01JB9819",
        "TN39AY5214",
        "KA01AB1234","KA02CD5678","KA03EF9012","KA04GH3456","KA05IJ7890",
        "MH12KL2345","DL08MN6789","KA50QR5678","KA53ST9012",
    ]

    # TN18AP6725 intentionally excluded — access denied for demo

    plates = list(set(plates))

    import random
    vehicle_types = ["Car", "Bike", "Scooty", "Commercial"]
    models = ["Maruti Swift", "Honda City", "Royal Enfield", "Hyundai Creta",
              "TVS Jupiter", "Bajaj Pulsar", "Tata Nexon", "KTM Duke",
              "Honda Activa", "Hero Splendor", "Yamaha FZ", "Maruti Baleno"]
    colors = ["White", "Black", "Silver", "Blue", "Red", "Grey", "Orange"]
    names = ["Rahul S", "Priya N", "Arun K", "Sneha R", "Vikram S",
             "Amit P", "Deepa V", "Karthik R", "Meera I", "Suresh B",
             "Lavanya G", "Rohan M", "Ananya D", "Nikhil R", "Siddharth T"]

    random.seed(42)
    sample_data = [
        (p, random.choice(names), random.choice(vehicle_types),
         random.choice(models), random.choice(colors))
        for p in plates
    ]

    c.executemany("""
        INSERT OR IGNORE INTO registered_vehicles
        (plate_number, owner_name, vehicle_type, vehicle_model, color)
        VALUES (?, ?, ?, ?, ?)
    """, sample_data)

    conn.commit()
    conn.close()
    print(f"[DB] Database initialized with {len(plates)} plates.")


def check_plate(plate_number: str):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("""
        SELECT * FROM registered_vehicles
        WHERE plate_number = ? AND is_active = 1
    """, (plate_number.upper().replace(" ", "").replace("-", ""),))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None


def log_access(plate_number: str, status: str, confidence: float, image_path: str = ""):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO access_logs (plate_number, status, confidence, image_path)
        VALUES (?, ?, ?, ?)
    """, (plate_number, status, confidence, image_path))
    conn.commit()
    conn.close()


def get_all_plates():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM registered_vehicles ORDER BY plate_number")
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_logs(limit: int = 50):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM access_logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def add_plate(plate_number, owner_name, vehicle_type, vehicle_model, color):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("""
            INSERT INTO registered_vehicles (plate_number, owner_name, vehicle_type, vehicle_model, color)
            VALUES (?, ?, ?, ?, ?)
        """, (plate_number.upper(), owner_name, vehicle_type, vehicle_model, color))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()