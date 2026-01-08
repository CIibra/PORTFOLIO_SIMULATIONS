# app_dash.py
import dash
from dash import dcc, html, Input, Output
import plotly.figure_factory as ff
import torch, numpy as np
from torchvision import models, transforms
from PIL import Image
import os, base64

classes = ["NORMAL","PNEUMONIA"]
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("model_resnet18.pt", map_location=device))
model = model.to(device); model.eval()

test_dir = "data/chest_xray/test"
items = []
for c in classes:
    cls_dir = os.path.join(test_dir, c)
    if os.path.exists(cls_dir):
        for f in sorted(os.listdir(cls_dir))[:50]:
            items.append({"path": os.path.join(cls_dir, f), "label": c})

app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("Analyse d’images médicales (Chest X-ray)"),

    dcc.Dropdown(
        id="img_select",
        options=[{"label": os.path.basename(i["path"]), "value": i["path"]} for i in items],
        placeholder="Choisissez une image test"
    ),

    # Texte centré au-dessus
    html.Div(id="pred_output", style={"textAlign":"center", "marginTop":"20px", "fontWeight":"bold"}),

    # Disposition côte à côte : image à gauche, matrice à droite
    html.Div([
        html.Div([
            html.Img(id="img_view", style={"maxWidth":"300px", "margin":"10px auto", "display":"block"})
        ], style={"flex":"1", "textAlign":"center"}),

        html.Div([
            dcc.Graph(id="conf_mat")
        ], style={"flex":"1"})
    ], style={"display":"flex", "flexDirection":"row", "justifyContent":"space-around", "marginTop":"20px"})
])

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

@app.callback(
    [Output("pred_output","children"), Output("img_view","src"), Output("conf_mat","figure")],
    Input("img_select","value")
)
def update(img_path):
    if not img_path:
        return "", "", {}
    probs = predict(img_path)
    pred_cls = classes[int(np.argmax(probs))]
    txt = f"Prédiction: {pred_cls} | NORMAL={probs[0]:.2f}, PNEUMONIA={probs[1]:.2f}"

    # Image affichée
    with open(img_path, "rb") as f:
        src = "data:image/png;base64," + base64.b64encode(f.read()).decode()

    # Matrice de confusion rapide
    from sklearn.metrics import confusion_matrix
    y_true, y_pred = [], []
    for i in items[:30]:
        p = predict(i["path"])
        y_true.append(classes.index(i["label"]))
        y_pred.append(int(np.argmax(p)))
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    fig = ff.create_annotated_heatmap(z=cm, x=classes, y=classes, colorscale="Blues")
    fig.update_layout(title="Matrice de confusion (échantillon)")

    return txt, src, fig

if __name__ == "__main__":
    app.run(debug=True)
