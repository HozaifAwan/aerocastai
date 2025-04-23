import csv

with open("daily_log.csv", "r") as infile:
    reader = csv.reader(infile)
    cleaned = [row for row in reader if len(row) == 16 and not any(x.startswith("<<<<") or x.startswith("====") or x.startswith(">>>>") for x in row)]

with open("daily_log.csv", "w", newline="") as outfile:
    writer = csv.writer(outfile)
    writer.writerows(cleaned)

print("✅ daily_log.csv cleaned successfully.")
