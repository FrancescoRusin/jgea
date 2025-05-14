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

import io.github.ericmedvet.jgea.core.distance.Edit;
import java.util.Map;
import java.util.Set;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToIntFunction;

public enum Base implements Type {

  BOOLEAN(
      Boolean.class,
      o -> 1,
      (o1, o2) -> o1.equals(o2) ? 0d : 1d
  ), INT(
      Integer.class,
      o -> 1,
      (o1, o2) -> {
        double i1 = ((Integer) o1).doubleValue();
        double i2 = ((Integer) o2).doubleValue();
        return Math.abs(i1 - i2);
      }
  ), REAL(
      Double.class,
      o -> 1,
      (o1, o2) -> Math.abs((Double) o1 - (Double) o2)
  ), STRING(
      String.class,
      o -> ((String) o).length(),
      (o1, o2) -> Edit.compute(
          ((String) o1).chars().boxed().toList(),
          ((String) o2).chars().boxed().toList()
      )
  );

  private final Class<?> javaClass;
  private final ToIntFunction<Object> sizer;
  private final ToDoubleBiFunction<Object, Object> dissimilarity;

  Base(Class<?> javaClass, ToIntFunction<Object> sizer, ToDoubleBiFunction<Object, Object> dissimilarity) {
    this.javaClass = javaClass;
    this.sizer = sizer;
    this.dissimilarity = dissimilarity;
  }

  @Override
  public boolean canTakeValuesOf(Type other) {
    return equals(other);
  }

  @Override
  public Type concrete(Map<Generic, Type> genericTypeMap) {
    return this;
  }

  @Override
  public Set<Generic> generics() {
    return Set.of();
  }

  @Override
  public boolean matches(Object o) {
    return javaClass.isInstance(o);
  }

  @Override
  public Map<Generic, Type> resolveGenerics(Type concreteType) {
    return Map.of();
  }

  @Override
  public int sizeOf(Object o) {
    return sizer.applyAsInt(o);
  }

  @Override
  public double dissimilarity(Object o1, Object o2) {
    return dissimilarity.applyAsDouble(o1, o2);
  }

  @Override
  public String toString() {
    return "%s".formatted(name().substring(0, 1));
  }

}
