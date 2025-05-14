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
package io.github.ericmedvet.jgea.problem.classification;

import io.github.ericmedvet.jgea.core.problem.SimpleEBMOProblem;
import io.github.ericmedvet.jgea.core.util.Misc;
import io.github.ericmedvet.jnb.datastructure.TriFunction;
import java.util.List;
import java.util.SequencedMap;
import java.util.Set;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;

public interface ClassificationProblem<X, Y extends Enum<Y>> extends SimpleEBMOProblem<Classifier<X, Y>, X, Y, ClassificationProblem.Outcome<Y>, Double> {
  record Outcome<Y>(Y actual, Y predicted) {}

  enum Metric implements Function<List<? extends Outcome<?>>, Double> {
    ERROR_RATE(
        outcomes -> (double) outcomes.stream()
            .filter(o -> !o.actual().equals(o.predicted()))
            .count() / (double) outcomes.size()
    ), WEIGHTED_ERROR_RATE(
        outcomes -> {
          Set<?> ys = outcomes.stream().map(Outcome::actual).collect(Collectors.toSet());
          return ys.stream()
              .mapToDouble(
                  y -> ERROR_RATE.function.apply(
                      outcomes.stream()
                          .filter(o -> o.actual().equals(y))
                          .toList()
                  )
              )
              .average()
              .orElseThrow();
        }
    );

    private final Function<List<? extends Outcome<?>>, Double> function;


    Metric(Function<List<? extends Outcome<?>>, Double> function) {
      this.function = function;
    }

    @Override
    public Double apply(List<? extends Outcome<?>> outcomes) {
      return function.apply(outcomes);
    }

    @Override
    public String toString() {
      return name().toLowerCase();
    }
  }

  List<Metric> metrics();

  @Override
  default SequencedMap<String, Objective<List<Outcome<Y>>, Double>> aggregateObjectives() {
    return metrics().stream()
        .collect(
            Misc.toSequencedMap(
                Metric::toString,
                m -> new Objective<>(m, Double::compareTo)
            )
        );
  }

  @Override
  default TriFunction<X, Y, Y, Outcome<Y>> errorFunction() {
    return (x, actual, predicted) -> new Outcome<>(actual, predicted);
  }

  @Override
  default BiFunction<Classifier<X, Y>, X, Y> predictFunction() {
    return Classifier::classify;
  }
}
