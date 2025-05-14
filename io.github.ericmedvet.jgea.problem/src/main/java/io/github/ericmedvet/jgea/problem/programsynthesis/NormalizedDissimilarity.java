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
import io.github.ericmedvet.jgea.core.util.IndexedProvider;
import io.github.ericmedvet.jnb.datastructure.DoubleRange;
import java.util.ArrayList;
import java.util.List;

public class NormalizedDissimilarity implements Distance<List<Object>> {

  private final Distance<List<Object>> rawDistance;
  private final DoubleRange rawRange;


  public NormalizedDissimilarity(
      List<Type> types,
      double maxDissimilarity,
      IndexedProvider<List<Object>> valueProvider
  ) {
    rawDistance = new Dissimilarity(types, maxDissimilarity);
    List<Double> dists = new ArrayList<>();
    for (int i = 0; i < valueProvider.size(); i++) {
      for (int j = 0; j < i; j++) {
        if (valueProvider.get(i) != null && valueProvider.get(j) != null) {
          dists.add(rawDistance.apply(valueProvider.get(i), valueProvider.get(j)));
        }
      }
    }
    rawRange = new DoubleRange(0, dists.stream().mapToDouble(d -> d).max().orElseThrow());
    if (rawRange.extent() == 0) {
      throw new IllegalArgumentException("Extent of dissimilarity range on cases is 0");
    }
  }

  @Override
  public Double apply(List<Object> os1, List<Object> os2) {
    if (os1 == null && os2 == null) {
      return 0d;
    }
    if (os1 == null || os2 == null) {
      return 2d;
    }
    return rawRange.normalize(rawDistance.apply(os1, os2));
  }
}
