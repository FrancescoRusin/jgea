/*-
 * ========================LICENSE_START=================================
 * jgea-problem
 * %%
 * Copyright (C) 2018 - 2024 Eric Medvet
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =========================LICENSE_END==================================
 */
package io.github.ericmedvet.jgea.problem.ca;

import io.github.ericmedvet.jnb.datastructure.Grid;
import io.github.ericmedvet.jsdynsym.core.numerical.MultivariateRealFunction;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class MultivariateRealGridCellularAutomaton extends GridCellularAutomaton<double[]> {

  public MultivariateRealGridCellularAutomaton(
      Grid<double[]> initialStates,
      List<Grid<Double>> convolutionKernels,
      MultivariateRealFunction updateFunction,
      boolean torodial) {
    super(
        initialStates,
        radiusFromKernels(convolutionKernels),
        convolutions(convolutionKernels).andThen(concatenator()).andThen(updateFunction),
        torodial,
        new double[initialStates.get(0, 0).length]);
  }

  private static int radiusFromKernels(List<Grid<Double>> kernels) {
    List<Integer> ws = kernels.stream().map(Grid::w).distinct().toList();
    List<Integer> hs = kernels.stream().map(Grid::h).distinct().toList();
    if (ws.size() != 1 || hs.size() != 1) {
      throw new IllegalArgumentException("Kernels are unconsistent in size: %s"
          .formatted(kernels.stream()
              .map(k -> "(%dx%d)".formatted(k.w(), k.h()))
              .collect(Collectors.joining(" "))));
    }
    int w = ws.get(0);
    int h = hs.get(0);
    if (w != h) {
      throw new IllegalArgumentException("Kernels are not squares: %dx%d".formatted(w, h));
    }
    if (w % 2 != 1) {
      throw new IllegalArgumentException("Kernels are not centered: %dx%d".formatted(w, h));
    }
    return (w - 1) / 2;
  }

  private static Function<Grid<double[]>, double[]> convolution(Grid<Double> kernel) {
    return g -> {
      if (g.w() != kernel.w() || g.h() != kernel.h()) {
        throw new IllegalArgumentException("Kernel and input sizes do not match: (%dx%d) vs. (%dx%d)"
            .formatted(kernel.w(), kernel.h(), g.w(), g.h()));
      }
      double[] out = new double[g.get(0, 0).length];
      kernel.entries().forEach(e -> {
        double[] in = g.get(e.key());
        IntStream.range(0, out.length).forEach(i -> out[i] = out[i] + e.value() * in[i]);
      });
      return out;
    };
  }

  private static Function<List<double[]>, double[]> concatenator() {
    return as -> {
      double[] out = new double[as.stream().mapToInt(a -> a.length).sum()];
      int c = 0;
      for (double[] a : as) {
        System.arraycopy(a, 0, out, c, a.length);
        c = c + a.length;
      }
      return out;
    };
  }

  private static Function<Grid<double[]>, List<double[]>> convolutions(List<Grid<Double>> kernels) {
    return g -> kernels.stream().map(k -> convolution(k).apply(g)).toList();
  }
}