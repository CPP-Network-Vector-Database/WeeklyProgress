import csv
from datetime import datetime
from sentence_transformers import SentenceTransformer
import os

# Load embedding model (384-dimensional)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Input CSV file path
csv_file = '/kaggle/input/5-tuple-ip-flow-data/ip_flow_dataset.csv'

# Output CSV file
output_csv_file = '/kaggle/working/processed3_ip_flow_data.csv'

def convert_time(time_str):
    """Convert timestamp string to ISO 8601 format with microsecond precision."""
    try:
        # Normalize spacing and remove timezone
        parts = time_str.strip().replace("  ", " ").rsplit(" ", 1)
        time_str_clean = parts[0]
        
        # Truncate to microseconds (6 digits after decimal)
        if '.' in time_str_clean:
            prefix, frac = time_str_clean.split('.')
            frac = frac[:6].ljust(6, '0')  # ensure 6 digits
            time_str_clean = f"{prefix}.{frac}"
        
        dt = datetime.strptime(time_str_clean, '%b %d, %Y %H:%M:%S.%f')
        return dt.isoformat()
    except Exception as e:
        print(f"Failed to parse time: {time_str} => {e}")
        return 'NULL'

def escape(val):
    """Escape string for CSV if needed."""
    return val.replace("'", "''")

def main():
    with open(csv_file, 'r') as infile, open(output_csv_file, 'w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = ['frame_number', 'frame_time', 'ip_src', 'ip_dst', 'tcp_srcport', 'tcp_dstport', 'protocol', 'frame_len', 'embedding']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        # Write header to the CSV file
        writer.writeheader()
        
        for row in reader:
            # Construct embedding input
            text_for_embedding = f"{row['ip.src']} {row['ip.dst']} {row['tcp.srcport']} {row['tcp.dstport']} {row['_ws.col.protocol']} {row['frame.len']}"
            vector = model.encode(text_for_embedding, show_progress_bar=False).tolist()
            
            # Prepare values
            frame_number = int(row['frame.number'])
            frame_time = convert_time(row['frame.time'])
            ip_src = row['ip.src']
            ip_dst = row['ip.dst']
            # Handle missing or invalid port values
            tcp_srcport = row['tcp.srcport'] if row['tcp.srcport'].isdigit() else 'NULL'
            tcp_dstport = row['tcp.dstport'] if row['tcp.dstport'].isdigit() else 'NULL'
            protocol = escape(row['_ws.col.protocol'])
            frame_len = int(row['frame.len'])
            vector_str = '[' + ','.join(f"{v:.6f}" for v in vector) + ']'

            # Write row to CSV
            writer.writerow({
                'frame_number': frame_number,
                'frame_time': f"{frame_time}+05:30",  # Add timezone if needed
                'ip_src': ip_src,
                'ip_dst': ip_dst,
                'tcp_srcport': tcp_srcport,
                'tcp_dstport': tcp_dstport,
                'protocol': protocol,
                'frame_len': frame_len,
                'embedding': vector_str
            })

    print(f"Generated processed CSV file: {output_csv_file}")

if __name__ == '__main__':
    main()