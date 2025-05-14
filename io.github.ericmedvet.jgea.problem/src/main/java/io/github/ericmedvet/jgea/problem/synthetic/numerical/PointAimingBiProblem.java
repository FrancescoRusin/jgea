/*-
 * ========================LICENSE_START=================================
 * jgea-problem
 * %%
 * Copyright (C) 2018 - 2025 Eric Medvet
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
package io.github.ericmedvet.jgea.problem.synthetic.numerical;

import io.github.ericmedvet.jgea.core.order.PartialComparator;
import io.github.ericmedvet.jgea.core.problem.SymmetricQualityBasedBiProblem;
import java.util.Collections;
import java.util.List;
import java.util.Optional;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.UnaryOperator;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class PointAimingBiProblem implements SymmetricQualityBasedBiProblem<List<Double>, List<Integer>, Double> {

  private final int p;
  private final double[] target;

  public PointAimingBiProblem(int p, double[] target) {
    this.p = p;
    this.target = target;
  }

  @Override
  public Optional<List<Double>> example() {
    return Optional.of(Collections.nCopies(p, 0d));
  }

  @Override
  public UnaryOperator<Double> symmetryFunction() {
    return q -> 1 - q;
  }

  @Override
  public BiFunction<List<Double>, List<Double>, List<Integer>> outcomeFunction() {
    return (s0, s1) -> IntStream.range(0, p)
        .mapToObj(i -> Math.abs(s0.get(i) - target[i]) < Math.abs(s1.get(i) - target[i]) ? 0 : 1)
        .collect(Collectors.toList());
  }

  @Override
  public Function<List<Integer>, Double> firstQualityFunction() {
    return o -> (double) o.stream().mapToInt(Integer::intValue).sum() / (double) p;
  }

  @Override
  public PartialComparator<Double> qualityComparator() {
    return PartialComparator.from(Double.class);
  }
}
