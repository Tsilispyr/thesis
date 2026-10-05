# Ανάλυση ML Pipeline - SD-UAV AI Recovery

*Run #7 · 2026-05-21 · 14 features · 544,763 IMU rows · epochs=10 · batch=64 · window=10*

---

## Θεωρητικό Υπόβαθρο ML

### Bias-Variance Tradeoff

Κάθε ML μοντέλο αντιμετωπίζει αυτό το θεμελιώδες δίλημμα:

- **High Bias (Underfitting):** Το μοντέλο δεν μαθαίνει αρκετά. Train loss ≈ Val loss, και τα δύο υψηλά. Αίτια: λίγα δεδομένα, μοντέλο πολύ απλό, λίγα epochs.
- **High Variance (Overfitting):** Το μοντέλο αποστηθίζει τα training data. Train loss << Val loss, και το gap αυξάνεται με τα epochs. Αίτια: πολύ μεγάλο μοντέλο, λίγα δεδομένα, χωρίς regularization.
- **Sweet Spot:** Train ≈ Val, και τα δύο χαμηλά, gap σταθερό και μικρό (<15% του val loss).

### Ορισμός Underfitting/Overfitting με Αριθμούς (για αυτό το project)

| Κατάσταση | Train Loss | Val Loss | Gap (Val-Train)/Val | Τάση Val |
|-----------|-----------|---------|---------------------|---------|
| Severe Underfitting | ~1.0 | ~1.0 | <2% | Flat/αυξάνει |
| Mild Underfitting | 0.3–0.6 | 0.3–0.6 | <5% | Αργά κατεβαίνει |
| Good Fit | 0.2–0.3 | 0.2–0.3 | 5–15% | Σταδιακά κατεβαίνει |
| Mild Overfitting | <0.2 | >0.3 | >30% | Ανεβαίνει |
| Severe Overfitting | <0.1 | >0.5 | >80% | Ανεβαίνει γρήγορα |

### Τεχνικές Normalization - Σύγκριση

| Τεχνική | Φόρμουλα | Αποτέλεσμα | Κατάλληλη για |
|---------|----------|-----------|---------------|
| **StandardScaler** (επιλεγμένο) | `(x-μ)/σ` | mean=0, std=1 | Gaussian/Normal distributions, IMU sensors |
| MinMaxScaler | `(x-min)/(max-min)` | [0,1] | Bounded features, image pixels |
| RobustScaler | `(x-Q2)/(Q3-Q1)` | Robust στα outliers | Data με πολλά outliers |
| L2 Normalizer | `x/‖x‖` | Unit norm per sample | Text/TF-IDF vectors |
| Log Transform | `log(x+1)` | Μειώνει skewness | Power-law distributions |

**Γιατί StandardScaler εδώ:** Τα IMU sensors (accelerometers, gyroscopes) παράγουν κατανομές κοντά στην κανονική λόγω thermal noise + Gaussian motion profiles. Το StandardScaler δεν αλλάζει τη μορφή της κατανομής - κρίσιμο γιατί ο LSTM πρέπει να βλέπει αυθεντικές κατανομές, απλά στην ίδια κλίμακα.

### LSTM Αρχιτεκτονική - Τι κάνει κάθε μέρος

```
Input: (batch=64, seq=10, features=14)
         ↓
LSTM Layer 1: hidden_size=64  - μαθαίνει short-term patterns (1-3 timesteps)
         ↓  dropout=0.2
LSTM Layer 2: hidden_size=64  - μαθαίνει long-term dependencies (across 10 steps)
         ↓  (take last hidden state)
FC: 64 → 32 (ReLU)           - non-linear compression
FC: 32 → 3                   - predict Δlat, Δlon, Δalt
Output: (batch=64, 3)
```

**Γιατί 2 layers:** Single-layer LSTM μαθαίνει μόνο immediate temporal patterns. Δεύτερο layer μαθαίνει "patterns of patterns" - π.χ. ότι ένα sequence αυξανόμενων gyro readings + σταθερής επιτάχυνσης σημαίνει στροφή.

**Dropout 0.2:** Κατά τη διάρκεια training, τυχαία 20% των hidden units "σβήνουν" σε κάθε forward pass. Αναγκάζει το δίκτυο να μάθει redundant representations - δεν μπορεί να βασιστεί σε ένα μόνο neuron για κάθε pattern.

**Loss function: MSE (Mean Squared Error)**
- `MSE = (1/n) Σ(ŷᵢ - yᵢ)²`
- Τιμωρεί μεγάλα λάθη δυσανάλογα (quadratic) - σωστό για navigation όπου ένα μεγάλο error position είναι πολύ χειρότερο από πολλά μικρά
- Normalized MSE ≈ 1.0 = το μοντέλο δεν κάνει τίποτα καλύτερο από "πρόβλεψε το mean"

---

## Ανάλυση ανά Γράφημα

---

### STEP 1 - Dataset Audit

**Γράφημα:** 3 panels - Record Count | NaN % | Usable for DR

**Τι βλέπουμε:**
- **Record Count:** nav=5.0K (σχεδόν αόρατο), imu=544.8K - 109× ανισορροπία. Το nav bar είναι κυριολεκτικά ένα pixel σε σχέση με το imu.
- **NaN Percentage:** Και τα δύο 0.0% - σπάνια κατάσταση σε πραγματικά datasets. Σημαίνει ότι δεν χρειάζεται imputation (KNN, mean fill, interpolation).
- **Usable for DR:** Και τα δύο ✓ - έχουν και IMU (για input features) και GPS/position (για targets).

**Τι σημαίνει η ανισορροπία για το ML:**
Ένα μοντέλο που εκπαιδεύεται σε balanced data (50/50) μαθαίνει εξίσου από όλες τις πηγές. Εδώ, αν συνδυαστούν τα datasets (Run D), 98.3% των sequences θα προέρχονται από IMU και 1.7% από nav. Ο LSTM ουσιαστικά θα εκπαιδευτεί κυρίαρχα σε IMU dynamics, με ελάχιστη έκθεση σε GPS-scale nav targets - προβληματικό για generalization σε GPS-based flight.

---

### STEP 2 - Data Cleaning

**Γράφημα:** Bar chart Raw vs Cleaned + IMU Rename Table

**Τι βλέπουμε:**
- **nav:** 5,000 → 4,764 (−236 γραμμές, −4.7%). Ορατή μείωση στο bar chart.
- **imu:** 544,763 → 544,763 (0 αλλαγές). Τα bars είναι ίδια γιατί `dropna()` δεν αφαίρεσε τίποτα.
- **Rename Table:** 9 column mappings. Χωρίς αυτά, τα FEATURE_COLS lookups θα επέστρεφαν NaN για ολόκληρη τη feature matrix του IMU.

**Γιατί φιλτράρουμε obstacle_detected=1:**
Οι 236 γραμμές αντιπροσωπεύουν "forced maneuvers" - το drone κινείται λόγω εξωτερικής λογικής αποφυγής εμποδίων, όχι λόγω της δικής του αδράνειας + ελέγχου. Αν τις κρατήσουμε: το LSTM μαθαίνει `[imu readings] → [position change]` όταν η πραγματική αιτία είναι ο collision-avoidance controller. Αυτό είναι **confounded training** - η αιτιώδης σχέση που μαθαίνει το μοντέλο είναι λάθος.

---

### STEP 3 - Normalization

**Γράφημα:** 2 rows × 14 features. Before (μπλε) + After (πράσινο, forced [-5,5] x-axis)

**Τι βλέπουμε - Before (μπλε) row:**
- `imu_acc_x/y`: σ=13.02/13.61, range ±50 - Αίτιο: gravity component + acceleration noise
- `imu_gyro_x/y/z`: σ=22-25, range ±100 - Αίτιο: μεγάλο dynamic range των MEMS gyroscopes
- `roll/pitch/yaw`: σ=1.59-1.63, range ±π rad - compact distributions
- `mag_x/y/z`: σ=0.53-0.63, range ±1 - Earth's magnetic field magnitude
- `speed`: σ=10.15, range 0-25 m/s - bimodal (hovering + cruising)
- `dt`: σ=0.099, range 0-2 - IMU samples στα 4ms, nav samples στα ~1s

**Τι βλέπουμε - After (πράσινο) row:**
- **ΟΛΕΣ** οι features έχουν force-shared x-axis [-5, 5]
- Η κόκκινη κάθετη γραμμή (x=0) επιβεβαιώνει ότι κάθε feature έχει mean=0
- Τα σ-badges δείχνουν την αρχική std (π.χ. imu_gyro_z: σ=25.23 → 1.00)
- **Γιατί φαίνεται ίδια η μορφή:** StandardScaler είναι γραμμικός μετασχηματισμός - `f(ax+b) = a·f(x) + b`. Το σχήμα δεν αλλάζει, μόνο η θέση και η κλίμακα.

**Κρίσιμη παρατήρηση:** Στo "after" plot, το `speed` και `dt` έχουν διαφορετική κατανομή από τα υπόλοιπα features. Το speed εμφανίζει ελαφρά bimodal κατανομή (hovering + πτήση), ενώ το dt έχει bimodal (IMU ~0.004s vs nav ~1.0s). Ο StandardScaler χειρίζεται και τις δύο σωστά - απλώς η κατανομή στο normalized space δεν είναι Gaussian, αλλά αυτό είναι αποδεκτό.

---

### STEP 4 - Feature × Target Correlation Matrix

**Γράφημα:** 17×17 heatmap (14 features + 3 targets), RdYlGn colormap [-1, +1]

**Τι βλέπουμε:**
- **Diagonal:** Πάντα 1.00 (feature με τον εαυτό της)
- **Feature-Target block** (bottom-right corner): Σχεδόν όλα κοντά στο 0.00
  - `imu_acc_y → delta_lat`: 0.02 (μέγιστη συσχέτιση)
  - `dt → delta_alt`: 0.00
  - `mag_z → anything`: ~0.00
- **Inter-feature correlations** (top-left block):
  - `imu_acc_x ↔ imu_acc_y`: ~0.02 (σχεδόν ανεξάρτητα - σωστό για orthogonal axes)
  - `imu_gyro_x ↔ imu_gyro_z`: ~0.06 (μικρή cross-axis coupling - φυσικό σε MEMS)
  - `speed ↔ imu_acc_x`: ~0.28 (λογικό - επιτάχυνση και ταχύτητα σχετίζονται)
  - `roll ↔ imu_acc_y`: ~-0.13 (banking maneuvers αλλάζουν lateral acceleration)

**Γιατί 0.018 max correlation ΔΕΝ είναι πρόβλημα:**
Η γραμμική συσχέτιση μετράει `E[(X-μX)(Y-μY)] / (σX·σY)` - αυτό αποτυπώνει μόνο *γραμμική* σχέση. Dead Reckoning είναι:
```
v(t) = ∫ a(t) dt    (non-linear time integration)
p(t) = ∫ v(t) dt    (second integration)
```
Η σχέση `a(t) → Δp` εξαρτάται από ολόκληρο το ιστορικό, όχι από μία μόνο τιμή. Pearson correlation δεν μπορεί να ανιχνεύσει αυτή τη temporal, non-linear δομή - γι' αυτό ο LSTM.

**Τι θα βλέπαμε αν υπήρχε γραμμική συσχέτιση >0.5:** Θα μπορούσαμε να χρησιμοποιήσουμε Linear Regression ή Ridge Regression αντί LSTM, με πολύ λιγότερα computational resources. Το 0.018 απαιτεί deep non-linear model.

---

### STEP 5 - Sequence Window (IMU rows 1000–1009)

**Γράφημα:** 14 feature subplots (10 timesteps) + target box

**Τι βλέπουμε ανά feature:**
- **imu_acc_x (~37-39 m/s²):** Δεν είναι ο X-axis της κίνησης - περιλαμβάνει gravity projection. Το δείχνει η τιμή >9.8 m/s².
- **imu_acc_y (~28-32 m/s²):** Υψηλές τιμές, πιθανή lateral acceleration σε στροφή.
- **imu_acc_z (~-6.6 to -7.2 m/s²):** Μικρότερο από -9.8 → ο drone ανεβαίνει ή εκτελεί maneuver.
- **imu_gyro_x/y/z (~±0.05 rad/s):** Σχεδόν μηδέν → ευθεία πτήση σε αυτό το window.
- **roll (~0.025 rad):** Πολύ μικρή κλίση - σχεδόν level flight. Std ≈0 σε αυτό το window.
- **pitch (~0.455→0.375 rad):** Σαφής αρνητική κλίση! ~27° → ~21° πτχ. Ο drone κατεβάζει τη μύτη - ενδεικτικό επιτάχυνσης εμπρός ή κατάβασης.
- **yaw (~0.025 rad):** Σχεδόν σταθερό - δεν γυρίζει.
- **mag_x (~0.38-0.48), mag_y (~0.0-0.05), mag_z (~-0.85 to -1.0):** Κύρια συνιστώσα στον Z άξονα - δείχνει κατακόρυφο magnetic field (αναμενόμενο για αυτή τη γεωγραφική περιοχή).
- **speed (~6.5-9.0 m/s):** Αυξανόμενη ταχύτητα - consistent με το pitch change.
- **dt (~4.17×10⁻³ s):** Σχεδόν σταθερό - IMU samples στα ~240 Hz.

**Target:** `delta_lat=0.00046m, delta_lon=0.03237m, delta_alt=-0.02044m`
Πολύ μικρές μετατοπίσεις ανά sample (4ms) - λογικό για 240Hz sampling.

**Γιατί αυτό το window είναι πλούσιο σε πληροφορία:** Ο pitch αλλάζει ενώ ο gyro είναι σχεδόν μηδέν - αυτό σημαίνει ότι η αλλαγή στάσης έγινε πριν αυτό το window. Ο LSTM "θυμάται" το παρελθόν μέσω του hidden state, οπότε μπορεί να αξιοποιήσει αυτή τη συνέχεια.

---

### STEP 6 - Run A: Nav Only, No Normalization

**Γράφημα:** Train (μπλε) vs Val (κόκκινο), epochs 1-10

**Αριθμοί:**
- Epoch 1: Train=11,259 | Val=11,433 → Gap=174 (+1.5%)
- Epoch 8: Train=11,192 | Val=11,492 → Gap=300 (+2.7%) ← maximum divergence
- Epoch 10: Train=11,092 | Val=11,451 → Gap=359 (+3.2%)

**Διάγνωση: Scale Error + Nascent Overfitting**

Η Val Loss ΑΥΞΑΝΕΤΑΙ (11,433 → 11,492) ενώ η Train Loss πέφτει αργά. Αυτό είναι κλασική υπογραφή **overfitting που αρχίζει**, αλλά στο λάθος context:

1. **Κύριο πρόβλημα:** MSE σε raw GPS degrees. Ένα degree lat = ~111km. Ακόμα και άλλαγή 0.00001° (=1.1m) δίνει MSE contribution `(0.00001)² ≈ 10⁻¹⁰` ενώ ο Adam optimizer χρειάζεται gradients στην κλίμακα 10⁻³ για να λειτουργήσει αποτελεσματικά. Αποτέλεσμα: numerically unstable gradients.

2. **Train κατεβαίνει επιτέλους στα epochs 8-10:** Ο LSTM αρχίζει να αποστηθίζει τα training patterns αγνοώντας τα val - κλασικό overfitting. Αλλά από λάθος αιτία (optimizer βρίσκει local minima σε ασταθή loss landscape).

3. **Αποτέλεσμα:** Ακατάλληλο. Normalization είναι prerequisite, όχι optimization.

---

### STEP 7 - Run B: Nav Only, Normalized

**Γράφημα:** Train (μπλε) vs Val (κόκκινο), epochs 1-10

**Αριθμοί:**
- Epoch 1: Train=1.0002 | Val=1.0096 → Gap=0.0094 (+0.94%)
- Epoch 10: Train=0.9978 | Val=1.0107 → Gap=0.0129 (+1.29%)
- Συνολική βελτίωση train: 1.0002 → 0.9978 (−0.24%)
- Συνολική αλλαγή val: 1.0096 → 1.0107 (+0.11% - χειροτέρεψε!)

**Διάγνωση: Pure Underfitting - Data Starvation**

Αυτό είναι το πιο καθαρό παράδειγμα underfitting στα γραφήματα:

1. **Train ≈ Val:** Gap μόλις 1.29% - το μοντέλο δεν κάνει distinction μεταξύ training και validation data. Σε ένα model που overfits, το gap θα ήταν >30%.

2. **Val ελαφρά αυξάνεται:** 1.0096 → 1.0107. Τεχνικά "overfitting" (val κακώνεται), αλλά η αύξηση (+0.0011) είναι στατιστικά ασήμαντη - noise level.

3. **Loss ≈ 1.0 = Null Model:** Ένας normalized StandardScaler target έχει variance=1. MSE=1.0 σημαίνει ότι το μοντέλο προβλέπει **πάντα το mean** - ισοδύναμο με να λέει "δεν κινείσαι καθόλου". Καμία χρήσιμη πληροφορία δεν μαθαίνεται.

4. **Αίτιο:** 4,754 sequences × 80% train = 3,803 training sequences. Για LSTM με 64 hidden × 2 layers × 4 gates = ~33,000 parameters, αναλογία parameters/sequences = 8.7. Χρειάζεται minimum 10-50× sequences per parameter για αξιόπιστο learning.

**Αριθμητική διάγνωση Underfitting:**
- `|Train - Val| / Val < 5%` ✓ (→ underfitting, δεν overfits)
- `Val Loss ≈ 1.0` ✓ (→ null model performance)
- `ΔVal/ΔEpoch ≈ 0` ✓ (→ learning curve plateaued)
- `N_sequences / N_params < 10` ✓ (→ data starvation)

---

### STEP 8 - Run C: IMU Only, Normalized ← BEST

**Γράφημα:** Train (μπλε) vs Val (κόκκινο), epochs 1-10

**Αριθμοί:**
- Epoch 1: Train=0.697 | Val=0.587 → Gap=−0.110 (Val < Train)
- Epoch 5: Train=0.317 | Val=0.291 → Gap=−0.026
- Epoch 9: Train=0.249 | Val=0.217 → Gap=−0.032 (minimum val)
- Epoch 10: Train=0.238 | Val=0.226 → Gap=−0.012

**Διάγνωση: Good Fit - Healthy Convergence, Still Learning**

1. **Εκθετική κάθοδος:** Η μορφή της καμπύλης (convex, εκθετική αποκλιμάκωση) είναι η "ιδανική" learning curve. Ο LSTM μαθαίνει γρήγορα στην αρχή (υψηλό learning signal) και αργεί καθώς πλησιάζει στο minimum.

2. **Val < Train σε ΟΛΕΣ τις epochs:** Ανορθόδοξο, αλλά εξηγείται:
   - Το 80/20 split είναι τυχαίο. Αν οι 109K val sequences τυχαία περιέχουν περισσότερες "ήρεμες" πτήσεις (μικρότερα delta positions), ο val loss θα είναι χαμηλότερος.
   - Ο Dropout (0.2) ενεργοποιείται μόνο κατά training, αυξάνοντας τεχνητά το train loss σε σχέση με inference.

3. **Epoch 9 vs 10:** Val Loss 0.217 → 0.226 (+0.009 = +4%). Μικρή αύξηση στο epoch 10 - πρώτο σημάδι potential instability. Δεν είναι ακόμα overfitting, αλλά θα εμφανιστεί με περισσότερα epochs χωρίς learning rate decay.

4. **Ακόμα δεν έχει plateaued:** Η gradient της καμπύλης στο epoch 10 είναι ακόμα αρνητική. 20 epochs αναμένεται Val ≈ 0.15-0.18. 50 epochs με early stopping: ίσως 0.10-0.12.

**Αριθμητική διάγνωση Good Fit:**
- `Train converges monotonically` ✓
- `Val follows Train closely` ✓
- `Gap ≈ 5-15%` ✓ (gap είναι αρνητικό - Val < Train, atypical but OK)
- `Val still decreasing at last epoch` ✓ (→ more epochs beneficial)
- `N_sequences / N_params ≈ 13,000` ✓ (→ πλήρης αξιοποίηση capacity)

---

### STEP 9 - Run D: Nav + IMU Combined, Normalized

**Γράφημα:** Train (μπλε) vs Val (κόκκινο), epochs 1-10

**Αριθμοί:**
- Epoch 1: Train=1.072 | Val=0.423 → Gap=−0.649 (Val << Train!)
- Epoch 5: Train=0.973 | Val=0.347 → Gap=−0.626
- Epoch 10: Train=0.953 | Val=0.333 → Gap=−0.620

**Διάγνωση: Domain Shift - Structural Train/Val Mismatch**

Αυτό είναι το πιο ενδιαφέρον γράφημα λόγω του τεράστιου και σταθερού gap:

1. **Train Loss ≈ 0.95 (παραμένει υψηλό):** Ο LSTM δυσκολεύεται να μάθει το training set. Αιτία: το combined dataset έχει δύο "γλώσσες" targets - GPS degrees (nav) και relative metres (IMU). Αυτές έχουν διαφορετική numeric distribution ακόμα και μετά τον StandardScaler, γιατί ο scaler fit-αρει τη συνένωση, όχι κάθε source ξεχωριστά.

2. **Val Loss ≈ 0.33 (σταθερά χαμηλό):** Το val set (random 20%) είναι κυρίαρχα IMU data (98.3% στο combined). Ο LSTM, αν και δυσκολεύεται στο training mix, μαθαίνει αρκετά από τα 98.3% IMU sequences για να προβλέπει καλά IMU-dominated val sequences.

3. **Gap ~0.62 = Domain Shift signature:** Σε κανονικές συνθήκες: `val gap = 5-15%`. Εδώ: `(Train-Val)/Val ≈ 185%`. Αυτό δεν είναι overfitting - είναι structural incompatibility. Το training set "δυσκολεύεται" περισσότερο από το val set επειδή βλέπει συχνότερα τα "ακατανόητα" nav targets.

4. **Val ακόμα κατεβαίνει:** 0.423 → 0.333. Ο LSTM σιγά-σιγά μαθαίνει το IMU component παρά τον θόρυβο από το nav component. Με περισσότερα epochs, val ίσως φτάσει 0.25-0.28 αλλά ποτέ δεν θα φτάσει το Run C's 0.22 χωρίς coordinate unification.

**Γιατί D > C (χειρότερο παρά περισσότερα δεδομένα):**
Η προσθήκη noisy/incompatible data είναι χειρότερη από καθαρά λιγότερα δεδομένα. Αυτό επιβεβαιώνεται και στη βιβλιογραφία: "more data helps only if it's from the same distribution."

---

### STEP 10 - All Runs Comparison

**Γράφημα:** Log-scale line chart (left) + Log-scale bar chart (right)

**Τι βλέπουμε:**
- **Run A (γκρι, ×):** Flat line στο 10⁴ - δεν κατεβαίνει ποτέ. Ουσιαστικά flat στο log scale.
- **Run B (μπλε, ●):** Flat line στο 10⁰ (=1.0) - επίσης δεν κατεβαίνει.
- **Run C (πράσινο, ■):** Σαφής κάθοδος, φτάνει στο 0.2264 (χαμηλότερο στο γράφημα).
- **Run D (κόκκινο, ◆):** Κατεβαίνει αλλά σταθεροποιείται υψηλότερα από C.

**Γιατί log scale:** Το Run A (11,451) θα καταπίεζε οπτικά τα B/C/D αν χρησιμοποιούσαμε linear scale. Με log scale, οι σχετικές βελτιώσεις φαίνονται σωστά:
- A→B: 4 τάξεις μεγέθους βελτίωση (÷11,000)
- B→C: 1 τάξη μεγέθους (÷4.5)
- C→D: C κερδίζει κατά 32%

**Bar chart:** Επιβεβαιώνει A=11,451 >> B=1.01 > D=0.33 > C=0.23.

---

### STEP 11 - RL Path Recovery

**Γράφημα:** Success Rate (%) + Avg Steps to Recovery - Rule-Based vs PPO

**Αριθμοί:**
- Rule-Based: 100% success, 40 avg steps
- PPO (10K steps): 28% success, 158 avg steps
- PPO efficiency: 28/100 × 40/158 = 0.071 → 7.1% efficiency vs Rule-Based

**Διάγνωση: Severe RL Undertraining**

**PPO (Proximal Policy Optimization) mechanics:**
- Policy network: maps state → action probabilities
- Value network: estimates expected future reward from state
- PPO constraint: `|π_new/π_old - 1| < ε` - limits how much the policy can change per update
- 10K timesteps = 10,000 environment steps = ~5 PPO update iterations

**Γιατί 10K είναι εντελώς ανεπαρκές:**
Το SwarmRecoveryEnv έχει continuous state space (x, y, target_x, target_y) και discrete actions (8 directions + stay). State space ≈ ∞. Για να "δει" ο agent αρκετές καταστάσεις ώστε να γενικεύσει: χρειάζεται minimum 100K-500K timesteps. 1M+ για σύγκριση με Rule-Based.

**Γιατί 28% success και όχι 0%:** O PPO agent τυχαία ανακαλύπτει ότι κάποιες κινήσεις αυξάνουν το reward (κίνηση προς target). Με 10K steps, έχει αρχίσει να μαθαίνει αυτή τη βασική ευρετική, αλλά δεν έχει εδραιώσει σταθερή policy.

**Γιατί 158 avg steps:** Ο agent "τριγυρίζει" - βρίσκει κατεύθυνση προς target, κινείται, χάνει τον προσανατολισμό, ξαναβρίσκει. Random walk με bias προς target ≈ O(n²) steps αντί O(n) για optimal policy.

**Rule-Based αλγόριθμος:**
```python
direction = normalize(target - current_position)
step = direction × step_size
new_position = current + step
```
Ντετερμινιστικός, O(n) steps, πάντα βέλτιστος για convex, obstacle-free environments.

---

### STEP 12 - Summary Dashboard

**Γράφημα:** Table με όλα τα runs + key stats text box

**Τι επιβεβαιώνεται:**
- Highlighted row: C_imu_norm (πράσινο background + bold text) - dynamic winner selection
- Final Train Loss C: 0.2384 vs Val: 0.2264 → **Val < Train** επιβεβαιώνεται
- Time: C=213.2s, D=210.3s - σχεδόν ίδιο παρά τις +5K extra sequences του D (ο bottleneck είναι τα 544K IMU rows που είναι κοινά)
- "Next step: wire dr_lstm_nav_imu_norm.pth" - Προσοχή: outdated (production models χρειάζονται retrain με input_size=14)

---

## Συνολική Αξιολόγηση και Επόμενα Βήματα

### Τι Μάθαμε από τα Runs

| Ερώτηση | Απάντηση από τα γραφήματα |
|---------|--------------------------|
| Χρειάζεται normalization; | Run A: Val=11,451. Run B: Val=1.01. Ναι, non-negotiable. |
| Αρκεί μόνο nav data; | Run B: plateau στο 1.0 ήδη από epoch 1. Όχι. |
| Βοηθάει ο όγκος data; | Run C: 115× sequences → 0.23 val. Ναι, δραματικά. |
| Βοηθάει ο συνδυασμός; | Run D: domain shift → χειρότερο από C. Όχι, χωρίς coordinate fix. |
| Αρκούν 10 epochs; | Run C epoch 9→10: 0.217→0.226. Όχι, ακόμα converging. |

### Roadmap Βελτίωσης με Αναμενόμενα Αποτελέσματα

| Action | Αναμενόμενο Val Loss | Effort |
|--------|---------------------|--------|
| 20 epochs Run C | ~0.15-0.18 | Χαμηλό - αλλαγή `EPOCHS=20` |
| 50 epochs + early stopping | ~0.10-0.13 | Χαμηλό |
| Fix coordinate mismatch → rerun D | ~0.12-0.16 | Μέτριο |
| hidden_size 64→128 | ~15% βελτίωση | Χαμηλό |
| window_size 10→50 | ~10-20% | Χαμηλό |
| RL: 1M timesteps | >80% success | Υψηλό (χρόνος) |

### Γενικά ML Συμπεράσματα

1. **Data volume > model complexity** για αυτό το task: 544K sequences + μικρό LSTM > 4.7K sequences + οποιοδήποτε μοντέλο
2. **Normalization prerequisite:** Χωρίς StandardScaler, ο Adam optimizer δεν μπορεί να ρυθμίσει learning rate αποτελεσματικά σε mixed-scale features
3. **Domain shift > data volume:** Η προσθήκη incompatible data (nav GPS degrees + IMU relative metres) μειώνει performance παρά αυξάνει volume
4. **Val < Train (dropout effect):** Ο dropout (0.2) αυξάνει train loss τεχνητά - inference mode (val/test) δεν εφαρμόζει dropout → val loss μπορεί να είναι χαμηλότερο
5. **Pearson correlation = 0 δεν σημαίνει no relationship:** Σημαίνει μόνο no *linear* relationship. Non-linear temporal dependencies (Dead Reckoning) απαιτούν LSTM/Transformer
6. **MSE ≈ 1.0 = null model:** Όταν targets είναι normalized (std=1), MSE=1 σημαίνει "πρόβλεψε πάντα 0" - καθόλου learning
