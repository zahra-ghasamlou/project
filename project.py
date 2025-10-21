import pandas as pd
import os
import numpy as np
from scipy.stats import ttest_ind
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.problem import ElementwiseProblem
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

data_path = r'C:\Users\admin\Downloads'

def load_and_preprocess_data():
    logging.info("Loading clinical and sheet data...")
    clinical_path = os.path.join(data_path, 'clinical.tsv')
    clinical_df = pd.read_csv(clinical_path, sep='\t')
    clinical_df.set_index('cases.submitter_id', inplace=True)

    sheet_path = os.path.join(data_path, 'gdc_sample_sheet.2025-09-29.tsv')
    sheet_df = pd.read_csv(sheet_path, sep='\t')
    file_to_case = dict(zip(sheet_df['File ID'], sheet_df['Case ID']))

    logging.info("Processing expression data...")
    expr_data_list = []
    processed = 0
    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        if os.path.isdir(folder_path) and len(folder_name) == 36 and '-' in folder_name[8:13]:
            file_uuid = folder_name
            tsv_path = None
            for file in os.listdir(folder_path):
                if file.endswith('.rna_seq.augmented_star_gene_counts.tsv'):
                    tsv_path = os.path.join(folder_path, file)
                    break
            if tsv_path:
                case_id = file_to_case.get(file_uuid)
                if case_id:
                    try:
                        expr_df = pd.read_csv(tsv_path, sep='\t', skiprows=1, low_memory=False)
                        expr_df = expr_df[(expr_df['gene_id'].str.startswith('ENSG', na=False)) & (expr_df['gene_type'] == 'protein_coding')]
                        expr_df = expr_df[['gene_id', 'tpm_unstranded']].copy()
                        expr_df['case_id'] = case_id
                        expr_df.rename(columns={'tpm_unstranded': 'expression'}, inplace=True)
                        expr_data_list.append(expr_df)
                        processed += 1
                        if processed % 50 == 0:
                            logging.info(f"Processed {processed} files")
                    except Exception as e:
                        logging.error(f"Error processing {file_uuid}: {e}")

    logging.info(f"Processed {processed} files")

    batch_size = 50
    full_list = []
    for i in range(0, len(expr_data_list), batch_size):
        batch = pd.concat(expr_data_list[i:i + batch_size], ignore_index=True)
        full_list.append(batch)
    expr_full = pd.concat(full_list, ignore_index=True)
    expr_long = expr_full.groupby(['case_id', 'gene_id'])['expression'].mean().reset_index()

    expr_wide = expr_long.pivot(index='case_id', columns='gene_id', values='expression')

    combined_df = clinical_df.join(expr_wide, how='inner')

    combined_path = os.path.join(data_path, 'combined_expression_clinical.csv')
    combined_df.to_csv(combined_path)

    luad_mask = combined_df['cases.primary_site'] == 'Bronchus and lung'
    luad_df = combined_df[luad_mask].copy()
    luad_unique = luad_df.groupby(level=0).first().reset_index()
    gene_cols = [col for col in luad_unique.columns if col.startswith('ENSG')]

    stage_col = 'diagnoses.ajcc_pathologic_stage'

    def get_stage_group(s):
        if pd.isna(s): return 'Unknown'
        s_str = str(s).upper().strip()
        if s_str == '--' or 'NAN' in s_str: return 'Unknown'
        stage_code = s_str.replace('STAGE ', '').strip().replace('-', '')
        early_stages = ['I', 'IA', 'IB', 'II', 'IIA', 'IIB']
        late_stages = ['III', 'IIIA', 'IIIB', 'IV', 'IVA', 'IVB']
        if stage_code in early_stages:
            return 'Early'
        elif stage_code in late_stages:
            return 'Late'
        return 'Unknown'

    luad_unique['stage_group'] = luad_unique[stage_col].apply(get_stage_group)
    valid_stage = luad_unique[luad_unique['stage_group'] != 'Unknown']

    luad_unique[gene_cols] = luad_unique[gene_cols].fillna(0)
    for col in gene_cols:
        luad_unique[col] = (luad_unique[col] - luad_unique[col].mean()) / (luad_unique[col].std() + 1e-8)

    luad_path = os.path.join(data_path, 'luad_final.csv')
    luad_unique.to_csv(luad_path)

    return valid_stage, gene_cols

def feature_selection_ttest(valid_stage, gene_cols):
    logging.info("Performing t-test with nested CV...")
    outer_kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_selected_genes = set()

    for outer_fold, (train_idx, test_idx) in enumerate(outer_kf.split(valid_stage)):
        train_outer = valid_stage.iloc[train_idx]
        inner_kf = KFold(n_splits=5, shuffle=True, random_state=42)
        inner_selected = set()
        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_kf.split(train_outer)):
            inner_train = train_outer.iloc[inner_train_idx]
            early_inner = inner_train[inner_train['stage_group'] == 'Early']
            late_inner = inner_train[inner_train['stage_group'] == 'Late']
            fold_genes = []
            for gene in gene_cols:
                early_vals = early_inner[gene].dropna()
                late_vals = late_inner[gene].dropna()
                if len(early_vals) > 10 and len(late_vals) > 10:
                    _, p_val = ttest_ind(early_vals, late_vals)
                    if p_val < 0.05:
                        fold_genes.append((gene, p_val))
            top15 = sorted(fold_genes, key=lambda x: x[1])[:15]
            inner_selected.update([g[0] for g in top15])
        all_selected_genes.update(inner_selected)
        logging.info(f"Outer fold {outer_fold + 1}: {len(inner_selected)} genes added")

    important_genes = list(all_selected_genes)
    logging.info(f"Important genes (union): {len(important_genes)}")

    genes_path = os.path.join(data_path, 'genes_important_luad.csv')
    pd.Series(important_genes).to_csv(genes_path, index=False)

    return important_genes

class NSGAIIProblem(ElementwiseProblem):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        super().__init__(n_var=X.shape[1], n_obj=2, xl=0, xu=1)

    def _evaluate(self, x, out, *args, **kwargs):
        selected_idx = np.where(x > 0.5)[0]
        if len(selected_idx) == 0:
            out["F"] = [1.0, 1.0]
            return
        X_sel = self.X[:, selected_idx]
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = cross_val_score(SVC(kernel='rbf', probability=True, random_state=42), X_sel, self.y, cv=cv, scoring='roc_auc')
        mean_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)
        out["F"] = [1 - mean_auc, std_auc]

def nsga2_self_tuning(valid_stage, important_genes):
    logging.info("Running NSGA2 with self-tuning...")
    X = valid_stage[important_genes].values
    y = (valid_stage['stage_group'] == 'Late').astype(int).values

    problem = NSGAIIProblem(X, y)
    algorithm = NSGA2(pop_size=40)
    res = minimize(problem, algorithm, ('n_gen', 50), seed=42, verbose=False)

    logging.info(f"NSGA2 completed: {len(res.X)} solutions in pareto front")

    ks = list(range(20, 101, 10))
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    svm_tune = SVC(kernel='rbf', probability=True, random_state=42)
    tune_results = []

    pareto_solutions = res.X
    pareto_fitness = res.F
    top_pareto_idx = np.argsort(np.sum(pareto_fitness, axis=1))[:10]

    for k in ks:
        best_auc_for_k = []
        for idx in top_pareto_idx:
            mask = pareto_solutions[idx] > 0.5
            num_selected = np.sum(mask)
            if abs(num_selected - k) <= 10:
                selected_idx = np.where(mask)[0][:k]
                if len(selected_idx) >= 10:
                    X_sel_k = X[:, selected_idx]
                    auc_scores_k = cross_val_score(svm_tune, X_sel_k, y, cv=cv, scoring='roc_auc')
                    mean_auc_k = auc_scores_k.mean()
                    best_auc_for_k.append(mean_auc_k)
        if best_auc_for_k:
            max_auc_k = max(best_auc_for_k)
            tune_results.append({'k': k, 'max_AUC': max_auc_k})
            logging.info(f"k={k}: max AUC {max_auc_k:.3f}")

    tune_df = pd.DataFrame(tune_results)
    if len(tune_df) > 0:
        optimal_k = tune_df.loc[tune_df['max_AUC'].idxmax(), 'k']
        optimal_auc = tune_df['max_AUC'].max()
    else:
        optimal_k = 60
        optimal_auc = 0.7

    logging.info(f"Optimal k: {optimal_k} with AUC {optimal_auc:.3f}")

    best_idx = top_pareto_idx[0]
    selected_mask = pareto_solutions[best_idx] > 0.5
    selected_genes = [important_genes[i] for i in np.where(selected_mask)[0]][:optimal_k]

    nsga_tune_path = os.path.join(data_path, 'tune_nsga_luad.csv')
    tune_df.to_csv(nsga_tune_path, index=False)
    genes_nsga_path = os.path.join(data_path, 'optimal_genes_nsga_luad.csv')
    pd.Series(selected_genes).to_csv(genes_nsga_path, index=False)

    return selected_genes

def classify_features(valid_stage, selected_genes):
    logging.info("Classifying features...")
    X_sel = valid_stage[selected_genes].values
    y = (valid_stage['stage_group'] == 'Late').astype(int).values

    classifiers = {
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'NB': GaussianNB(),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
    }

    class SOAE(nn.Module):
        def __init__(self, input_dim):
            super(SOAE, self).__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Linear(32, 16))
            self.decoder = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, input_dim))
            self.classifier = nn.Linear(16, 1)

        def forward(self, x):
            enc = self.encoder(x)
            return self.classifier(enc), self.decoder(enc)

    def train_soae(X_train, y_train, epochs=50):
        model = SOAE(X_train.shape[1])
        optimizer = optim.Adam(model.parameters())
        criterion_class = nn.BCEWithLogitsLoss()
        criterion_recon = nn.MSELoss()
        for epoch in range(epochs):
            optimizer.zero_grad()
            class_out, recon = model(torch.tensor(X_train, dtype=torch.float32))
            loss = criterion_class(class_out.squeeze(), torch.tensor(y_train, dtype=torch.float32)) + criterion_recon(recon, torch.tensor(X_train, dtype=torch.float32))
            loss.backward()
            optimizer.step()
        return model

    def predict_soae(model, X):
        with torch.no_grad():
            class_out, _ = model(torch.tensor(X, dtype=torch.float32))
        return (torch.sigmoid(class_out.squeeze()).numpy() > 0.5).astype(int), torch.sigmoid(class_out.squeeze()).numpy()

    def compute_metrics(y_true, y_pred, y_prob=None):
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sn = tp / (tp + fn) if (tp + fn) > 0 else 0
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0
        return {'Acc': acc, 'F1': f1, 'AUC': auc, 'Sn': sn, 'Sp': sp}

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    results = []
    for train_idx, test_idx in kf.split(X_sel):
        X_train, X_test = X_sel[train_idx], X_sel[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        fold_results = {}
        for name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1] if hasattr(clf, 'predict_proba') else None
            fold_results[name] = compute_metrics(y_test, y_pred, y_prob)
        soae_model = train_soae(X_train, y_train)
        y_pred_soae, y_prob_soae = predict_soae(soae_model, X_test)
        fold_results['SOAE'] = compute_metrics(y_test, y_pred_soae, y_prob_soae)
        results.append(fold_results)

    avg_results = []
    for name in list(classifiers.keys()) + ['SOAE']:
        metrics = {metric: np.mean([r[name][metric] for r in results]) for metric in ['Acc', 'F1', 'AUC', 'Sn', 'Sp']}
        avg_results.append({'Classifier': name, **metrics})

    avg_df = pd.DataFrame(avg_results).round(3)
    logging.info("Average Metrics:\n" + avg_df.to_string())

    table3_path = os.path.join(data_path, 'results_table3_luad.csv')
    avg_df.to_csv(table3_path, index=False)

    return avg_df

def map_genes_to_pathways(selected_genes):
    logging.info("Mapping genes to pathways...")
    optimal_genes_clean = [g.split('.')[0] for g in selected_genes]

    mapping_txt_path = os.path.join(data_path, 'mart_export.txt')
    mapping_df = pd.read_csv(mapping_txt_path, sep=',', low_memory=False)

    mapping_df['Gene stable ID clean'] = mapping_df['Gene stable ID'].str.split('.').str[0]
    mapping_df = mapping_df[mapping_df['Reactome ID'].notna() & (mapping_df['Reactome ID'] != '')]

    optimal_df = pd.DataFrame({'Gene_stable_ID_clean': optimal_genes_clean})
    mapped = optimal_df.merge(mapping_df, left_on='Gene_stable_ID_clean', right_on='Gene stable ID clean', how='left')
    pathways = mapped['Reactome ID'].dropna().unique().tolist()

    logging.info(f"Pathways mapped: {len(pathways)}")

    pathway_path = os.path.join(data_path, 'pathways_from_genes.csv')
    pd.Series(pathways).to_csv(pathway_path, index=False)

    return pathways, mapped

def association_rule_mining(valid_stage, selected_genes, pathways, mapped):
    logging.info("Performing ARM on pathways...")
    # Gene to pathway mapping
    gene_to_path = mapped.groupby('Gene stable ID clean')['Reactome ID'].apply(list).to_dict()

    X_genes = valid_stage[selected_genes].fillna(0).values

    pathway_scores = np.zeros((len(valid_stage), len(pathways)))
    for i, pathway in enumerate(pathways):
        genes_in_path = [gene for gene, paths in gene_to_path.items() if pathway in paths]
        gene_indices = [selected_genes.index(g) for g in genes_in_path if g in selected_genes]
        if gene_indices:
            pathway_scores[:, i] = X_genes[:, gene_indices].mean(axis=1)

    scaler = MinMaxScaler()
    pathway_scores_norm = scaler.fit_transform(pathway_scores)

    data_arm = pd.DataFrame(pathway_scores_norm, columns=pathways)
    data_arm['stage_group'] = valid_stage['stage_group'].values

    discretized = pd.DataFrame()
    for col in pathways:
        bins = [-np.inf] + list(data_arm[col].quantile([0.33, 0.66])) + [np.inf]
        labels = [f'{col}_low', f'{col}_medium', f'{col}_high']
        discretized[col] = pd.cut(data_arm[col], bins=bins, labels=labels)

    discretized['stage_group'] = data_arm['stage_group']

    transactions = discretized.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df_trans = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = fpgrowth(df_trans, min_support=0.3, use_colnames=True, max_len=4)
    rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.1)

    logging.info(f"Number of rules: {len(rules)}")

    early_rules = rules[rules['consequents'].apply(lambda x: 'Early' in str(x))]
    late_rules = rules[rules['consequents'].apply(lambda x: 'Late' in str(x))]

    early_path = set()
    for ant in early_rules['antecedents']:
        for item in ant:
            if '_' in item:
                p = item.split('_')[0]
                early_path.add(p)

    late_path = set()
    for ant in late_rules['antecedents']:
        for item in ant:
            if '_' in item:
                p = item.split('_')[0]
                late_path.add(p)

    logging.info(f"Top 3 early pathways: {list(early_path)[:3]}")
    logging.info(f"Top 3 late pathways: {list(late_path)[:3]}")

    rules_path = os.path.join(data_path, 'arm_rules_luad.csv')
    rules.to_csv(rules_path)

    return rules

if __name__ == "__main__":
    valid_stage, gene_cols = load_and_preprocess_data()
    important_genes = feature_selection_ttest(valid_stage, gene_cols)
    selected_genes = nsga2_self_tuning(valid_stage, important_genes)
    avg_df = classify_features(valid_stage, selected_genes)
    pathways, mapped = map_genes_to_pathways(selected_genes)
    rules = association_rule_mining(valid_stage, selected_genes, pathways, mapped)
    logging.info("Simulation completed successfully!")