import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go


def fn_gen_plotly_bar(df, x_col, y_col,
                      txt_col=None, color_col=None, v_h='h', margin=None,
                      title=None, height=None, width=None, op=None, barmode=None,
                      legend=True, lg_title=None, lg_pos=None, lg_x=None, lg_top=True, x_range=None,
                      showtick_y=True):

    fig = px.bar(df, x=x_col, y=y_col, orientation=v_h, title=title, text=txt_col, color=color_col,
                 width=width, height=height, opacity=op)

    fig.update_traces(textfont_size=12)
    fig.update_layout(margin=margin,
                      xaxis_title='',
                      yaxis_title='',
                      xaxis_range=x_range,
                      yaxis=go.YAxis(showticklabels=showtick_y),
                      width=width,
                      height=height,
                      showlegend=legend,
                      barmode=barmode,
                      font=dict(
                          # family="Courier New, monospace",
                          size=14,
                          ),
                      # legend=dict(
                      #     title=lg_title,
                      #     orientation=lg_pos,
                      #     yanchor="bottom",
                      #     y=1.02,
                      #     xanchor="right",
                      #     x=lg_x),
                      )
    if lg_top:
        fig.update_layout(legend=dict(
                              title=lg_title,
                              orientation=lg_pos,
                              yanchor="bottom",
                              y=1.02,
                              xanchor="right",
                              x=lg_x))

    return fig
