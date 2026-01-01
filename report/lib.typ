#import "@local/zero:0.6.0": format-table, num, zi
#import "@preview/lilaq:0.5.0" as lq

#let benchmark-table(data) = {
  show: format-table(auto, auto, auto, auto, auto, auto, auto)
  let map = lq.color.map.cork
  let grd = gradient.linear(
    lq.color.map.okabe-ito.at(1),
    lq.color.map.okabe-ito.at(2),
  )
  table(
    columns: 7,
    table.header(
      $N$,
      $alpha$,
      $beta$,
      $rho$,
      [Baseline (Greedy)],
      [New Cost],
      [Improvement],
    ),
    ..for r in data.results {
      (
        num[#r.num_cities],
        num[#r.alpha],
        num[#r.beta],
        num[#r.density],
        num(
          exponent: (sci: 2),
          round: (mode: "figures", precision: 2),
        )[#r.baseline_cost],
        num(
          exponent: (sci: 2),
          round: (mode: "figures", precision: 2),
        )[#r.solution_cost],
        {
          set text(grd.sample(r.improvement_percent * 1%))
          [#num(round: (mode: "figures", precision: 2))[#r.improvement_percent]%]
        },
      )
    },
  )
}
