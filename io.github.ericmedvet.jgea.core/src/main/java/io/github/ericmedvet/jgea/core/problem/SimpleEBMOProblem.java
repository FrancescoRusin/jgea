/*-
 * ========================LICENSE_START=================================
 * jgea-core
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
package io.github.ericmedvet.jgea.core.problem;

import io.github.ericmedvet.jgea.core.util.Misc;
import java.util.Comparator;
import java.util.List;
import java.util.SequencedMap;
import java.util.function.Function;

public interface SimpleEBMOProblem<S, EI, EO, EQ, O> extends EBMOProblem<S, EI, EO, EQ, SequencedMap<String, O>, O>, SimpleMOProblem<S, O> {

  SequencedMap<String, Objective<List<EQ>, O>> aggregateObjectives();

  @Override
  default Function<List<EQ>, SequencedMap<String, O>> aggregateFunction() {
    return eqs -> Misc.sequencedTransformValues(aggregateObjectives(), obj -> obj.function().apply(eqs));
  }

  @Override
  default SequencedMap<String, Comparator<O>> comparators() {
    return Misc.sequencedTransformValues(aggregateObjectives(), Objective::comparator);
  }

}
