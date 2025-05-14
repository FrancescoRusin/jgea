/*-
 * ========================LICENSE_START=================================
 * jgea-core
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

package io.github.ericmedvet.jgea.core.representation.programsynthesis.type;

import java.util.Map;
import java.util.Set;

public interface Type {

  boolean canTakeValuesOf(Type other);

  Type concrete(Map<Generic, Type> genericTypeMap);

  double dissimilarity(Object o1, Object o2);

  Set<Generic> generics();

  boolean matches(Object o);

  Map<Generic, Type> resolveGenerics(Type concreteType) throws TypeException;

  int sizeOf(Object o);

  default boolean isGeneric() {
    return !generics().isEmpty();
  }

}
