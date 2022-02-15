import pandas as pd
import plotly.express as px
# import plotly.graph_objects as go

csvfile = "D:/YOLOX/YOLOX_outputs/yolox_landing_platform_nano/run-.-tag-Train_iou loss.csv"


def train_response():
    csv = pd.read_csv(csvfile)
    return csv


df = train_response()
fig = px.line(df, x="Step", y="Value")
# fig.show()


# fig = go.Figure(go.Scatter(x=df['Step'], y=df['Value'], name=""))
fig.update_xaxes(showline=True, linewidth=1, linecolor='black',
                 showgrid=True, gridwidth=0.25, gridcolor='gray',
                 tickfont_family="Arial Black")
fig.update_yaxes(showline=True, linewidth=1, linecolor='black',
                 showgrid=True, gridwidth=0.25, gridcolor='gray',
                 tickfont_family="Arial Black")
fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                  plot_bgcolor='rgba(0,0,0,0)',
                  width=1600,
                  height=800,
                  showlegend=True,
                  xaxis_title="Epoch",
                  yaxis_title="IOU loss",
                  yaxis=dict(
                      dtick=0.4
                  ),
                  xaxis=dict(
                    tick0=0,
                    dtick=25
                  ),
                  font=dict(
                      size=35,
                      family="Arial Black",
                  )
                  )
fig.update_traces(
    line=dict(
        width=5
    )
)

fig.show()
