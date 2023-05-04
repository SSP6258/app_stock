# import plotly.graph_objs as go
import plotly.express as px
import plotly.graph_objects as go
import os
from PIL import Image


def fn_show_img(IMG_file):
    img = IMG_file if IMG_file.endswith('.gif') or IMG_file.startswith('http') else Image.open(IMG_file)
    return img


def fn_gen_plotly_line(df, x, y, op=1, color='red', line_width=3):

    fig = px.line(df, x=x, y=y, markers=True)
    fig.update_traces(line_color=color, line_width=line_width, opacity=op)

    return fig


def fn_gen_plotly_bar(df, x_col, y_col,
                      txt_col=None, color_col=None, colors=None, v_h='h', margin=None,
                      title=None, height=None, width=None, op=None, barmode=None,
                      legend=True, lg_title=None, lg_pos=None, lg_x=None, lg_top=True, x_range=None,
                      showtick_y=True, showscale=True, text_auto=False, textposition='inside', color_mid=None,
                      showspike=False):

    fig = px.bar(df, x=x_col, y=y_col, orientation=v_h, title=title, text=txt_col, color=color_col,
                 width=width, height=height, opacity=op, text_auto=text_auto, color_continuous_midpoint=color_mid)

    fig.for_each_trace(lambda t: t.update(text=[]) if '勝率_new' not in t.name else ())

    fig.update_traces(textfont_size=14, textposition=textposition, cliponaxis=False, marker_color=colors)

    fig.update_layout(margin=margin,
                      xaxis_title='',
                      yaxis_title='',
                      xaxis_range=x_range,
                      yaxis=go.layout.YAxis(showticklabels=showtick_y),
                      width=width,
                      height=height,
                      showlegend=legend,
                      coloraxis_showscale=showscale,
                      barmode=barmode,
                      font=dict(size=14),
                      legend=dict(font_size=14,
                                  title=lg_title),
                      )

    if lg_top:

        fig.update_layout(legend=dict(
                              font_size=14,
                              title=lg_title,
                              orientation=lg_pos,
                              yanchor="bottom",
                              y=1.02,
                              xanchor="right",
                              x=lg_x),)

    if showspike:
        fig.update_xaxes(showspikes=True, spikecolor="grey", spikesnap="cursor", spikemode="across", spikethickness=1,
                         spikedash='solid')

        fig.update_yaxes(showspikes=True, spikecolor="grey", spikesnap="cursor", spikemode="across", spikethickness=1,
                         spikedash='solid')

    return fig
