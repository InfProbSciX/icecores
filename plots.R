
library(ggplot2)
library(data.table)
library(reticulate)

py_run_string("  # py_run_string('from IPython import embed; embed()')

import numpy as np
from utils import load_data

depth, data, expert_labels, eruptions = load_data()

")

np_load = import('numpy')$load

plot_df = data.table(
    depth=py$depth,
    data=py$data,
    hmm_dis=np_load('dis_batch.npy'),
    expert=py$expert_labels
)
plot_df[np_load('single_pass_0_idx.npy') + 1, hmm_ties := np_load('single_pass_0.npy')]
plot_df[np_load('single_pass_1_idx.npy') + 1, hmm_ties := np_load('single_pass_1.npy')]

for(i in 1:nrow(py$eruptions))
    plot_df[which.min(abs(py$eruptions[i, 1] - depth)), tie := py$eruptions[i, 2]]

plot_df[, year_col := as.factor(expert %% 2)]

ggplot(plot_df, aes(x=depth)) +
    geom_line(aes(y=hmm_dis, color='Dis-HMM (batches)'), alpha=0.5) +
    geom_line(aes(y=hmm_ties, color='Dis-HMM (w/ ties)'), alpha=0.5) +
    geom_point(aes(y=tie, shape='Volcanic Obs'), size=2) +
    geom_line(aes(y=expert, color='Expert Labels'), alpha=0.5) +
    labs(x='depth (m)', y='year') +
    theme_classic() +
    guides(shape=guide_legend(override.aes=list(size=0.5)),
           color=guide_legend(override.aes=list(size=0.5))) +
    theme(legend.title=element_text(size=9),
          legend.text=element_text(size=7),
          legend.position=c(0.25, 0.33),
          legend.key.size=unit(0.5, "lines"))

ggsave(file="comparison.pdf", width=6, height=4, dpi=300)

plot_df_data = plot_df[,
    .(depth = depth[c(1, .N)], label=c('min', 'max')),
    by=.(expert, year_col)
]
plot_df_data = dcast(plot_df_data, expert + year_col ~ label, value.var='depth')

ggplot() +
    geom_rect(aes(xmin=min, xmax=max, ymin=-Inf, ymax=Inf, fill=year_col), alpha=0.4, data=plot_df_data) +
    geom_line(aes(x=depth, y=data), data=plot_df) +
    labs(x='depth (m)', y='log concentration') +
    scale_fill_brewer(palette='Blues') +
    guides(fill="none") +
    theme_classic() +
    theme(axis.ticks.y=element_blank(), axis.text.y=element_blank()) +
    xlim(0, 19)

ggsave(file="data.pdf", width=5, height=2, dpi=300)
