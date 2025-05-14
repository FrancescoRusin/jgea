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

package io.github.ericmedvet.jgea.problem.synthetic;

import io.github.ericmedvet.jgea.core.problem.SimpleMOProblem;
import io.github.ericmedvet.jgea.core.representation.sequence.integer.IntString;
import io.github.ericmedvet.jgea.core.util.Misc;
import java.util.*;
import java.util.function.Function;
import java.util.stream.IntStream;

public record MultiObjectiveIntOneMax(
    SequencedMap<String, Comparator<Double>> comparators,
    Function<IntString, SequencedMap<String, Double>> qualityFunction,
    Optional<IntString> example
) implements SimpleMOProblem<IntString, Double> {

  public MultiObjectiveIntOneMax(int p, int upperBound) {
    this(
        IntStream.range(1, upperBound)
            .boxed()
            .collect(
                Misc.toSequencedMap(
                    MultiObjectiveIntOneMax::objectiveName,
                    i -> Double::compareTo
                )
            ),
        is -> IntStream.range(1, upperBound)
            .boxed()
            .collect(
                Misc.toSequencedMap(
                    MultiObjectiveIntOneMax::objectiveName,
                    i -> 1d - (double) is.genes()
                        .stream()
                        .filter(gi -> gi.equals(i))
                        .count() / (double) is.size()
                )
            ),
        Optional.of(new IntString(Collections.nCopies(p, 0), 0, upperBound))
    );
  }

  private static String objectiveName(int i) {
    return "obj%2d".formatted(i);
  }
}
