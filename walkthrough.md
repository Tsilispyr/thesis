# Walkthrough: Project Restructuring & Workspace Consolidation

Ολοκληρώθηκε με επιτυχία η συγχώνευση και η αναδιοργάνωση του project! Έχουμε πλέον ένα καθαρό και ενιαίο workspace, με το [AI_Recovery](file:///d:/SAR/latest/SAR-SIMULATION/AI_Recovery) να αποτελεί τον κεντρικό φάκελο λειτουργίας του project σου.

## Η Νέα Δομή
Όλος ο κώδικας, τα δεδομένα και ο προσομοιωτής βρίσκονται πλέον οργανωμένα στο `AI_Recovery`:

```text
AI_Recovery/
├── ai_backend/             # Όλος ο κώδικας της Python (API, Models, Data Processing)
│   ├── api/                # udp_server.py, path_recovery.py, swarm_logic.py
│   ├── models/             # rl_path_recovery.py, dead_reckoning_model.py
│   ├── data_processing/    # dataset_parser.py, osm_terrain_fetcher.py
│   ├── train.py
│   ├── eval_metrics.py
│   ├── Dockerfile
│   └── requirements.txt
├── datasets/               # Τα CSV αρχεία (μεταφέρθηκαν εδώ)
├── simulation/             # Ολοκληρωμένο το Godot Project (μεταφέρθηκε από το DRONE\droneSIM)
├── docker-compose.yml      # Ενημερωμένο orchestrator
└── README.md               # User guide
```

## Αλλαγές Συνδεσιμότητας
- **Δεδομένα:** Τα scripts `train.py` και `dataset_parser.py` διαβάζουν πλέον απροβλημάτιστα τα CSVs από τον νέο τοπικό φάκελο `datasets/`.
- **Docker:** Το `docker-compose.yml` ρυθμίστηκε ώστε να χτίζει το image διαβάζοντας μέσα από τον φάκελο `ai_backend/` και να δεσμεύει τα UDP ports (14551, 14552) σωστά.
- Τα παλιά αρχεία και φάκελοι (`Backend`, `Frontend`, `DRONE` κ.λπ.) στο root `SAR-SIMULATION` παρέμειναν άθικτα ως backup, αλλά πλέον δεν παίζουν κανένα ρόλο. Μπορείς να τα διαγράψεις όποτε νιώσεις έτοιμος.

## Πώς να ξεκινήσεις
1. **Για τον Server (μέσω Docker)**: Τρέξε `docker-compose up --build` μέσα από τον φάκελο `AI_Recovery`.
2. **Για τα Python scripts**: Τρέξε π.χ. `python ai_backend/eval_metrics.py`.
3. **Για το Godot**: Άνοιξε το Godot Engine και κάνε Import το project.godot που βρίσκεται στο `AI_Recovery/simulation/project.godot`. 

Όλα είναι έτοιμα και διασυνδεδεμένα στο νέο τους "σπίτι"!
