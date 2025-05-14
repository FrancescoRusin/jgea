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
package io.github.ericmedvet.jgea.problem.programsynthesis;

import io.github.ericmedvet.jgea.core.distance.Distance;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Type;
import java.util.List;
import java.util.stream.IntStream;

public class Dissimilarity implements Distance<List<Object>> {
  private final List<Type> types;
  private final double maxDissimilarity;

  public Dissimilarity(List<Type> types, double maxDissimilarity) {
    this.types = types;
    this.maxDissimilarity = maxDissimilarity;
  }

  @Override
  public Double apply(List<Object> os1, List<Object> os2) {
    if (os1 == null && os2 == null) {
      return 0d;
    }
    if (os1 == null) {
      return maxDissimilarity;
    }
    if (os2 == null) {
      return maxDissimilarity;
    }
    if (os1.size() != types.size() || os2.size() != types.size()) {
      throw new IllegalArgumentException(
          "Wrong sizes: %d expected, %d and %d found".formatted(
              types.size(),
              os1.size(),
              os2.size()
          )
      );
    }
    return Math.min(
        maxDissimilarity,
        IntStream.range(0, types.size())
            .mapToDouble(i -> types.get(i).dissimilarity(os1.get(i), os2.get(i)))
            .average()
            .orElse(0d)
    );
  }
}
