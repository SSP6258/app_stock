import plotly.graph_objs as go
import plotly.express as px


def fn_gen_plotly_bar(df, x_col, y_col,
                      txt_col=None, color_col=None, v_h='h', margin=None,
                      title=None, height=None, width=None, op=None, barmode=None,
                      legend=True, lg_title=None, lg_pos=None, lg_x=None):

    fig = px.bar(df, x=x_col, y=y_col, orientation=v_h, title=title, text=txt_col, color=color_col,
                 width=width, height=height, opacity=op)

    fig.update_layout(margin=margin,
                      showlegend=legend,
                      barmode=barmode,
                      legend=dict(
                          title=lg_title,
                          orientation=lg_pos,
                          x=lg_x)
                      )

    return fig
