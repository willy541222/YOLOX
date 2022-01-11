import pandas as pd
import plotly.graph_objects as go

csvfile = "D:/YOLOX/YOLOX_outputs/yolox_landing_platform_nano/run-.-tag-val_AP50_95.csv"


def train_response():
    csv = pd.read_csv(csvfile)
    return csv


df = train_response()
# fig = px.line(df, x="Epoch", y="Loss", title='YOLOX_nano Landing Platform Training Response Total Loss',)
# fig.show()


fig = go.Figure(go.Scatter(x=df['Step'], y=df['Value'], name=""))

fig.update_layout(plot_bgcolor='rgb(230, 230,230)',
                  showlegend=True,
                  xaxis_title="Epoch",
                  yaxis_title="Mean Average Precision 50:95",
                  yaxis=dict(
                      dtick=0.05
                  ),
                  xaxis=dict(
                    tick0=0,
                    dtick=25
                  ),
                  font=dict(
                      size=28,
                      family="Times New Roman"
                  )
                  )

fig.show()
