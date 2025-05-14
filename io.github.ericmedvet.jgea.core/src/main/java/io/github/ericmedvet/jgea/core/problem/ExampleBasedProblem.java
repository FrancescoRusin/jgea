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

import io.github.ericmedvet.jnb.datastructure.TriFunction;
import java.util.function.BiFunction;

public interface ExampleBasedProblem<S, EI, EO, EQ, Q> extends CaseBasedProblem<S, ExampleBasedProblem.Example<EI, EO>, EQ, Q> {
  record Example<I, O>(I input, O output) {}

  TriFunction<EI, EO, EO, EQ> errorFunction();

  BiFunction<S, EI, EO> predictFunction();

  @Override
  default BiFunction<S, Example<EI, EO>, EQ> caseFunction() {
    return (s, e) -> errorFunction().apply(e.input, e.output, predictFunction().apply(s, e.input));
  }
}
