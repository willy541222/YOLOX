import pandas as pd
import plotly.express as px
# import plotly.graph_objects as go

csvfile = "D:/YOLOX/YOLOX_outputs/yolox_slp_adam/run-.-tag-val_AP50.csv"


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
                  height=1200,
                  showlegend=True,
                  xaxis_title="Epoch",
                  yaxis_title="Average Precision 50",
                  yaxis_range=[0, 1],
                  yaxis=dict(
                      dtick=0.2
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
