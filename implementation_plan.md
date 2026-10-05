# Πλάνο Υλοποίησης: Project Restructuring & Consolidation

Βάσει του αιτήματός σου, θα δημιουργήσουμε ένα καθαρό και ενοποιημένο workspace με βάση τον φάκελο `AI_Recovery`. Οτιδήποτε άχρηστο (από το UGV project σε Frontend/Backend) μένει εκτός, και ό,τι χρειαζόμαστε έρχεται μέσα, για να έχουμε ένα "ready-to-run" project.

## 1. Νέα Δομή Φακέλων (Μέσα στο `AI_Recovery`)

Η νέα δομή του `AI_Recovery` θα είναι η εξής:
```
AI_Recovery/
├── ai_backend/             # Ο κώδικας της Python (API, Models, Data Processing)
├── simulation/             # Το Godot Project (μεταφορά από DRONE\droneSIM)
├── datasets/               # Τα CSV αρχεία (μεταφορά από το root datasets)
├── docker-compose.yml      # Το κεντρικό docker orchestrator
└── README.md               # Οδηγός Χρήσης
```

## 2. Ενέργειες που θα γίνουν (Αυτόματα μέσω terminal)
1. **Μεταφορά Δεδομένων**: Αντιγραφή του φακέλου `datasets` μέσα στο `AI_Recovery`.
2. **Μεταφορά Προσομοιωτή**: Αντιγραφή ολόκληρου του Godot project `DRONE\droneSIM` μέσα στο `AI_Recovery\simulation`.
3. **Αναδιοργάνωση Python Code**: Μετακίνηση των `api/`, `models/`, `data_processing/`, `train.py`, `eval_metrics.py`, `requirements.txt` και `Dockerfile` μέσα σε έναν υποφάκελο `ai_backend/` για απόλυτη καθαριότητα.

## 3. Αλλαγές στον Κώδικα (Connectivity Fixes)
Για να μην "σπάσει" η συνδεσιμότητα λόγω των νέων φακέλων, θα γίνουν οι εξής αλλαγές:
1. **Paths Δεδομένων (Python)**: Στο `dataset_parser.py` και στο `train.py`, τα paths θα αλλάξουν για να δείχνουν στον τοπικό φάκελο `../datasets` αντί για το root του δίσκου.
2. **Docker-Compose**: Το `docker-compose.yml` θα ενημερωθεί ώστε να κάνει build από τον υποφάκελο `ai_backend`.
3. **Imports**: Διόρθωση των python `sys.path.append()` για να βρίσκουν σωστά τα modules.

## User Review Required
> [!IMPORTANT]
> 1. **Διαγραφή Παλιών Φακέλων:** Θέλεις, αφού γίνει η μεταφορά, να **διαγράψω** τους παλιούς φακέλους (`Backend`, `Frontend`, `DRONE`, `datasets`) από το `d:\SAR\latest\SAR-SIMULATION` για να καθαρίσει τελείως ο δίσκος, ή απλά να τους αγνοούμε και να δουλεύουμε μόνο μέσα στο `AI_Recovery`;
> 2. Συμφωνείς με τη νέα δενδρική δομή που προτείνω παραπάνω;

Παρακαλώ επιβεβαίωσε για να ξεκινήσω τη μεταφορά αρχείων και τη διόρθωση των paths!
