import os

script_dir = os.path.dirname(os.path.abspath(__file__))
base_labels_dir = os.path.join(script_dir, "dataset\\labels")

for split in ("train", "val"):
    labels_dir = os.path.join(base_labels_dir, split)
    if not os.path.isdir(labels_dir):
        print(f"Directory not found, skipping: {labels_dir}")
        continue

    for filename in os.listdir(labels_dir):
        if not filename.endswith(".txt"):
            continue

        filepath = os.path.join(labels_dir, filename)
        corrected_lines = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # línea inválida, la saltamos

                try:
                    cls = int(parts[0]) - 1  # restamos 1 a la clase
                except ValueError:
                    continue

                if cls < 0:
                    print(f"Warning: {filepath}:{line_no} class became negative, skipping line")
                    continue

                rest = parts[1:]
                corrected_lines.append(f"{cls} " + " ".join(rest))

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(corrected_lines))

print("Todas las clases en labels/train y labels/val han sido corregidas (clase = clase - 1).")
