import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from shiny import reactive
from shiny.express import input, render, ui
from shinywidgets import render_plotly
import plotly.express as px
import matplotlib.pyplot as plt

# rdkit
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import py3Dmol


from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    roc_auc_score,
)

from rdkit.Chem import QED
from rdkit import Chem
from rdkit.Chem import RDConfig
import os, sys


# RDKit Contrib/SA_Score 경로를 찾아서 추가
contrib_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
sys.path.append(contrib_path)
import sascorer


###
from shiny import run_app

if __name__ == "__main__":
    # 'app'은 app.py 내에 정의된 App 객체 이름입니다.
    # 필요에 따라 'app.py:app' 또는 'your_module:app' 형태로 지정하세요.
    run_app(
        "app", 
        host="0.0.0.0",  # 모든 인터페이스 바인딩
        port=8000,       # 원하는 포트로 설정
        reload=True      # 코드 변경 시 자동 재시작
    )
####

# Morgan fingerprint generator
fpgen = AllChem.GetMorganGenerator(radius=3, fpSize=1024)

# ─── 페이지 설정 ────────────────────────────────────────────
ui.page_opts(title="AIDD", fillable=True)

# ─── 사이드바 ──────────────────────────────────────────────
with ui.sidebar(open="desktop"):
    @render.ui
    def side_controls():
        tab = input.tab()
        if tab== "SMILES Preview":
            return [
                ui.input_text("SMILES_text","Input SMILES")
            ]
        elif tab == "Data Preprocessing":
            return [
                ui.input_file("file", "Pick a CSV file", accept=".csv"),
            ]
        elif tab == "Modeling":
            return [
                ui.h3("Model Configuration"),
                ui.input_numeric("n_layers", "Hidden layers", value=2, min=1, max=5, step=1),
                ui.input_text("units", "Units per layer (comma-separated)", value="64,64"),
                ui.input_select(
                    "act_fn", "Activation function",
                    choices=["ReLU", "LeakyReLU", "Sigmoid", "Tanh"],
                    selected="ReLU"
                ),
                ui.input_checkbox("use_bn", "Use BatchNorm1d", value=False),
                ui.input_slider("dropout", "Dropout rate", min=0.0, max=0.5, step=0.05, value=0.0),
                ui.input_select(
                    "loss_fn", "Loss function",
                    choices=["MSELoss", "BCEWithLogitsLoss"],
                    selected="MSELoss"
                ),
                ui.input_select("opt", "Optimizer", choices=["SGD", "Adam"], selected="Adam"),
                ui.input_slider("lr", "Learning rate", min=1e-5, max=1e-1, step=1e-5, value=1e-3),
                ui.input_numeric("n_epochs", "Epochs", value=100, min=1, max=1000, step=1),
                ui.input_numeric("batch_size", "Batch size", value=64, min=1, max=1024, step=1),
                ui.input_action_button("train", "Train Model"),
            ]

        elif tab == "Virtual Screening":
            return [
                ui.input_file("file_c", "Pick a Candidate CSV file", accept=".csv"),
                # ui.input_slider("sim_thresh", "Similarity Threshold",
                #                 min=0.0, max=1.0, step=0.01, value=0.8),
                # ui.input_numeric("pred_min", "Predicted Min", value=0.0, step=0.01),
                # ui.input_numeric("pred_max", "Predicted Max", value=1.0, step=0.01),
                # ui.input_slider("qed_range", "QED Range",
                #                 min=0.0, max=1.0, value=(0.0,1.0), step=0.01),
                # ui.input_slider("sa_range", "SA Range",
                #                 min=0.0, max=10.0, value=(0.0,10.0), step=0.1),
            ]
        else:
            return []



# ─── 본문: 탭 네비게이션 ────────────────────────────────────
with ui.navset_pill(id="tab"):
    # SMILES Preview 탭
    with ui.nav_panel("SMILES Preview"):
        with ui.layout_columns(col_widths=[6, 6]):
            # 2D 구조 카드
            with ui.card(full_screen=True):
                ui.card_header("2D Structure")
                @render.ui
                def show_2d():
                    mol_h, _ = preview_mols()
                    if not mol_h:
                        return None
                    from rdkit.Chem.Draw import MolToImage
                    import io, base64
                    img = MolToImage(mol_h, size=(300,300))
                    buf = io.BytesIO()
                    img.save(buf, format='PNG')
                    data = base64.b64encode(buf.getvalue()).decode('ascii')
                    return ui.HTML(f"<img src='data:image/png;base64,{data}' width='300' height='300'/>")
            # 3D 구조 카드
            with ui.card(full_screen=True):
                ui.card_header("3D Structure")
                @render.ui
                def show_3d():
                    _, mol_3d = preview_mols()
                    if not mol_3d:
                        return None
                    view = py3Dmol.view(width=300, height=300)
                    view.addModel(Chem.MolToMolBlock(mol_3d), 'mol')
                    view.setStyle({'stick': {}})
                    view.zoomTo()
                    html = view._make_html()
                    return ui.HTML(html)
        # 분자 속성 카드
        with ui.card(full_screen=True):
            ui.card_header("Molecular Properties")
            @render.data_frame
            def props_table():
                mol_h, _ = preview_mols()
                if not mol_h:
                    return pd.DataFrame()
                from rdkit.Chem.Descriptors import (
                    MolWt, MolLogP, NumHAcceptors, NumHDonors,
                    FractionCSP3, NumRotatableBonds, RingCount,
                    TPSA, NumAromaticRings
                )
                props = {
                    "MW": MolWt(mol_h),
                    "LogP": MolLogP(mol_h),
                    "HBA": NumHAcceptors(mol_h),
                    "HBD": NumHDonors(mol_h),
                    "CSP3": FractionCSP3(mol_h),
                    "NumRotBond": NumRotatableBonds(mol_h),
                    "NumRings": RingCount(mol_h),
                    "TPSA": TPSA(mol_h),
                    "NumAromaticRings": NumAromaticRings(mol_h),
                    "SAS": sascorer.calculateScore(mol_h),
                    "QED": QED.default(mol_h)
                }
                return pd.DataFrame([props])
            

    # ─ Panel A ──────────────────────────────────────────────
    with ui.nav_panel("Data Preprocessing"):
        with ui.layout_columns(col_widths=[8, 4], height="200px"):
            with ui.card(full_screen=True):
                ui.card_header("Data Preview")
                @render.data_frame
                def table():
                    return render.DataGrid(df_data())
            with ui.card(full_screen=True):
                ui.card_header("Descriptive Statistics")
                @render.data_frame
                def desc_table():
                    df = df_data()
                    return render.DataGrid(df.describe()) if not df.empty else render.DataGrid(pd.DataFrame())
        with ui.layout_columns(col_widths=[12], height="300px"):
            with ui.card(full_screen=True):
                ui.card_header("Targets Histogram")
                @render_plotly
                def hist():
                    df = df_data()
                    if df.empty:
                        import plotly.graph_objects as go
                        return go.Figure(layout={"title": "No data"})
                    fig = px.histogram(df, x="Targets", nbins=30)
                    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                    return fig
        @render.text
        def feat_summary_plain():
            X, y = features()
            if X is None:
                return "No features generated yet."
            return (
                f"✔️ Generated fingerprints\n"
                f"  • X tensor shape: {tuple(X.shape)}\n"
                f"  • y tensor shape: {tuple(y.shape)}"
            )

    # ─ Panel B ──────────────────────────────────────────────
    with ui.nav_panel("Modeling"):
        with ui.layout_columns(col_widths=[12]):
            @render.text
            def model_summary_text():
                return str(model_def())
            @render.text
            def train_params_text():
                return (
                    f"Learning Rate: {input.lr():.5f}\n"
                    f"Epochs: {int(input.n_epochs())}\n"
                    f"Batch Size: {int(input.batch_size())}"
                )
        with ui.layout_columns(col_widths=[12], height="300px"):
            with ui.card(full_screen=True):
                ui.card_header("Training Loss Curve")
                @render.plot
                def loss_plot():
                    train_loss, _ = fit()
                    fig, ax = plt.subplots()
                    ax.plot(train_loss, label="Train Loss")
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title("DNN Model Loss")
                    ax.legend()
                    return fig
        with ui.layout_columns(col_widths=[12]):
            @render.data_frame
            def metrics_table():
                _, metrics = fit()
                if not metrics:
                    return pd.DataFrame()
                if input.loss_fn() == "BCEWithLogitsLoss":
                    # 분류 모델일 때
                    return pd.DataFrame([{
                        "Test Loss":      round(metrics["test_loss"], 3),
                        "AUC":            round(metrics["auc"], 3),
                        "Accuracy":       round(metrics["accuracy"], 3),
                        "Precision":      round(metrics["precision"], 3),
                        "Recall":         round(metrics["recall"], 3),
                        "F1-score":       round(metrics["f1_score"], 3),
                        "Balanced Acc.":  round(metrics["balanced_accuracy"], 3),
                    }])
                else:
                    # 회귀 모델일 때
                    return pd.DataFrame([{
                        "Test Loss": round(metrics["test_loss"], 3),
                        "MSE":       round(metrics["mse"], 3),
                        "MAE":       round(metrics["mae"], 3),
                        "RMSE":      round(metrics["rmse"], 3),
                        "R²":        round(metrics["r2_score"], 3),
                    }])

    # ─ Panel C ──────────────────────────────────────────────
    
    import tempfile  # <- 추가

    # Virtual Screening
    with ui.nav_panel("Virtual Screening"):
        with ui.card(full_screen=True):
            ui.card_header("Filtered Candidates")
            @render.data_frame
            def candidate_table():
                df = candidate_data()
                return render.DataGrid(df)
            
            @render.download(label="Download CSV")
            def download_candidates():
                # DataFrame 가져오기
                df = candidate_data()
                tmp_path = os.path.join(tempfile.gettempdir(), "filtered_candidates.csv")
                df.to_csv(tmp_path, index=False)
                return tmp_path




# ─── Reactive 계산부 ───────────────────────────────────────
@reactive.Calc
def preview_mols():
    smi = input.SMILES_text()
    if not smi:
        return None, None
    mol = Chem.MolFromSmiles(smi)
    mol_h = Chem.AddHs(mol)
    mol_h_3d = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xf10d
    AllChem.EmbedMolecule(mol_h_3d, params)
    return mol_h, mol_h_3d




@reactive.Calc
def df_data():
    files = input.file()
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(
        files[0]["datapath"], sep=None, engine="python", usecols=["Smiles", "pChEMBL Value"]
    )
    df = df.dropna(subset=["pChEMBL Value"]).drop_duplicates()
    return df.rename(columns={"pChEMBL Value": "Targets"})

@reactive.Calc
def features():
    df = df_data()
    if df.empty:
        return None, None
    fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in df['Smiles']]
    X = torch.tensor([list(fp) for fp in fps], dtype=torch.float32)
    y = torch.tensor(df['Targets'].to_numpy(dtype=float), dtype=torch.float32)
    return X, y

@reactive.Calc
def model_def():
    dims = [1024] + [int(x) for x in input.units().split(",")][: input.n_layers()] + [1]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            if input.use_bn():
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append({
                "ReLU": nn.ReLU(),
                "LeakyReLU": nn.LeakyReLU(),
                "Sigmoid": nn.Sigmoid(),
                "Tanh": nn.Tanh(),
            }[input.act_fn()])
            dr = input.dropout()
            if dr > 0:
                layers.append(nn.Dropout(dr))
    return nn.Sequential(*layers)

@reactive.Calc
def training_params():
    loss = {"MSELoss": nn.MSELoss(), "BCEWithLogitsLoss": nn.BCEWithLogitsLoss()}[input.loss_fn()]
    opt_cls = {"SGD": torch.optim.SGD, "Adam": torch.optim.Adam}[input.opt()]
    return loss, opt_cls(model_def().parameters(), lr=input.lr())

@reactive.Calc
def fit():
    input.train()
    X, y = features()
    if X is None:
        return [], {}

    # train/test split
    train_X, test_X, train_y, test_y = train_test_split(
        X.numpy(), y.numpy(), test_size=0.1, shuffle=True, random_state=42
    )

    model = model_def()
    loss_fn, optimizer = training_params()

    # ─── Training Loop ──────────────────────────────────
    train_loss = []
    bs = int(input.batch_size())
    for _ in range(int(input.n_epochs())):
        model.train()
        batch_losses = []
        for i in range(0, len(train_X), bs):
            xb = torch.tensor(train_X[i : i + bs], dtype=torch.float32)
            yb = torch.tensor(train_y[i : i + bs], dtype=torch.float32)
            logits = model(xb).squeeze()
            l = loss_fn(logits, yb)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            batch_losses.append(l.item())
        train_loss.append(float(np.mean(batch_losses)))

    # ─── Evaluation ────────────────────────────────────
    model.eval()
    with torch.no_grad():
        xb_test = torch.tensor(test_X, dtype=torch.float32)
        logits = model(xb_test).squeeze()

        if input.loss_fn() == "BCEWithLogitsLoss":
            # 분류 지표 계산 (타깃을 FloatTensor로)
            probs = torch.sigmoid(logits).cpu().numpy()
            y_true = test_y.astype(int)
            y_pred = (probs >= 0.5).astype(int)

            test_loss = float(
                loss_fn(
                    logits,
                    torch.tensor(y_true, dtype=torch.float32, device=logits.device),
                ).item()
            )

            metrics = {
                "test_loss":        test_loss,
                "auc":              roc_auc_score(y_true, probs),
                "accuracy":         accuracy_score(y_true, y_pred),
                "precision":        precision_score(y_true, y_pred, zero_division=0),
                "recall":           recall_score(y_true, y_pred, zero_division=0),
                "f1_score":         f1_score(y_true, y_pred, zero_division=0),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
            }

        else:
            # 회귀 지표 계산
            preds = logits.cpu().numpy()
            test_loss = float(
                loss_fn(
                    torch.tensor(preds, dtype=torch.float32, device=logits.device),
                    torch.tensor(test_y, dtype=torch.float32, device=logits.device),
                ).item()
            )

            mse = mean_squared_error(test_y, preds)
            metrics = {
                "test_loss": test_loss,
                "mse":        mse,
                "mae":        mean_absolute_error(test_y, preds),
                "rmse":       np.sqrt(mse),
                "r2_score":   r2_score(test_y, preds),
            }

    return train_loss, metrics



@reactive.Calc
def candidate_data():
    files = input.file_c()
    if not files:
        return pd.DataFrame()
    candidates = pd.read_csv(files[0]["datapath"])
    train_fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in df_data()["Smiles"]]
    from rdkit import DataStructs
    sims = [
        max((DataStructs.TanimotoSimilarity(fp, tf) for tf in train_fps), default=0)
        for fp in [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in candidates["smiles"]]
    ]
    mask = [s < input.sim_thresh() for s in sims]
    filtered = candidates.loc[mask].copy()
    Xcand = torch.stack([
        torch.tensor(list(fp), dtype=torch.float32)
        for fp, m in zip(
            [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in candidates["smiles"]],
            mask
        ) if m
    ])
    model = model_def()
    model.eval()
    with torch.no_grad():
        preds = model(Xcand).squeeze().numpy()
    filtered["Predicted"] = np.round(preds, 3)
    filtered["QED"] = filtered["smiles"].apply(lambda smi: round(QED.default(Chem.MolFromSmiles(smi)), 3))
    filtered["SA"] = filtered["smiles"].apply(lambda smi: round(sascorer.calculateScore(Chem.MolFromSmiles(smi)), 3))
    return filtered.sort_values("Predicted").reset_index(drop=True)
