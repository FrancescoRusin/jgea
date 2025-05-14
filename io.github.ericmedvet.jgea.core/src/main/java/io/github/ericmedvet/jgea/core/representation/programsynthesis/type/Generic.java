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

public record Generic(String name) implements Type {
  public static Generic of(String name) {
    return new Generic(name);
  }

  @Override
  public boolean canTakeValuesOf(Type other) {
    return true;
  }

  @Override
  public Type concrete(Map<Generic, Type> genericTypeMap) {
    Type concrete = genericTypeMap.get(this);
    if (concrete == null) {
      return this;
    }
    if (concrete.equals(this)) {
      return this;
    }
    if (concrete.generics().contains(this)) {
      return concrete;
    }
    return concrete.concrete(genericTypeMap);
  }

  @Override
  public double dissimilarity(Object o1, Object o2) {
    throw new UnsupportedOperationException("Cannot compute dissimilarity of generic type");
  }

  @Override
  public Set<Generic> generics() {
    return Set.of(this);
  }

  @Override
  public boolean matches(Object o) {
    return true;
  }

  @Override
  public Map<Generic, Type> resolveGenerics(Type concreteType) {
    return Map.of(this, concreteType);
  }

  @Override
  public int sizeOf(Object o) {
    throw new UnsupportedOperationException("Cannot compute size of generic type");
  }

  @Override
  public String toString() {
    return name;
  }
}
