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
package io.github.ericmedvet.jgea.core.solver.bi;

import io.github.ericmedvet.jgea.core.Factory;
import io.github.ericmedvet.jgea.core.order.PartialComparator;
import io.github.ericmedvet.jgea.core.problem.QualityBasedBiProblem;
import io.github.ericmedvet.jgea.core.solver.AbstractPopulationBasedIterativeSolver;
import io.github.ericmedvet.jgea.core.solver.Individual;
import io.github.ericmedvet.jgea.core.solver.POCPopulationState;
import java.util.List;
import java.util.function.BinaryOperator;
import java.util.function.Function;
import java.util.function.Predicate;

public abstract class AbstractBiEvolver<T extends POCPopulationState<I, G, S, Q, P>, P extends QualityBasedBiProblem<S, O, Q>, I extends Individual<G, S, Q>, G, S, Q, O> extends AbstractPopulationBasedIterativeSolver<T, P, I, G, S, Q> {

  protected final BinaryOperator<Q> fitnessReducer;

  public AbstractBiEvolver(
      Function<? super G, ? extends S> solutionMapper,
      Factory<? extends G> genotypeFactory,
      Predicate<? super T> stopCondition,
      boolean remap,
      BinaryOperator<Q> fitnessReducer,
      List<PartialComparator<? super I>> additionalIndividualComparators
  ) {
    super(solutionMapper, genotypeFactory, stopCondition, remap, additionalIndividualComparators);
    this.fitnessReducer = fitnessReducer;
  }
}
