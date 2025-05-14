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
package io.github.ericmedvet.jgea.problem.synthetic;

import io.github.ericmedvet.jgea.core.problem.SimpleMOProblem;
import io.github.ericmedvet.jgea.core.util.Misc;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public record LettersMax(
    Function<String, SequencedMap<String, Double>> qualityFunction,
    SequencedMap<String, Comparator<Double>> comparators,
    Optional<String> example
) implements SimpleMOProblem<String, Double> {

  public LettersMax(
      SequencedSet<String> letters,
      int l
  ) {
    this(
        s -> countCharOccurrences(s, letters),
        letters.stream()
            .collect(
                Misc.toSequencedMap(
                    letter -> ((Comparator<Double>) Double::compareTo).reversed()
                )
            ),
        Optional.of(String.join("", Collections.nCopies(l, ".")))
    );
  }

  private static SequencedMap<String, Double> countCharOccurrences(String s, SequencedSet<String> letters) {
    SequencedMap<String, Integer> rawCount = new LinkedHashMap<>(
        s.codePoints()
            .mapToObj(c -> String.valueOf((char) c))
            .collect(
                Collectors.groupingBy(
                    c -> c,
                    Collectors.summingInt(c -> 1)
                )
            )
    );
    letters.forEach(l -> rawCount.computeIfAbsent(l, ll -> 0));
    return Misc.sequencedTransformValues(rawCount, c -> (double) c / (double) s.length());
  }

}
