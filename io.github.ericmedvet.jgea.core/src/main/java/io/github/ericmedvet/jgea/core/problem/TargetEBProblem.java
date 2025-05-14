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

import io.github.ericmedvet.jgea.core.util.IndexedProvider;
import java.util.function.Function;

public interface TargetEBProblem<S, EI, EO, EQ, Q> extends ExampleBasedProblem<S, EI, EO, EQ, Q> {
  Function<? super EI, ? extends EO> target();

  IndexedProvider<EI> inputProvider();

  IndexedProvider<EI> validationInputProvider();

  @Override
  default IndexedProvider<Example<EI, EO>> caseProvider() {
    return inputProvider().then(ei -> new Example<>(ei, target().apply(ei)));
  }

  @Override
  default IndexedProvider<Example<EI, EO>> validationCaseProvider() {
    return validationInputProvider().then(ei -> new Example<>(ei, target().apply(ei)));
  }
}
