/*-
 * ========================LICENSE_START=================================
 * jgea-experimenter
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
package io.github.ericmedvet.jgea.experimenter.builders;

import io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gate;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn.Gates;
import io.github.ericmedvet.jgea.core.representation.tree.numeric.Element;
import io.github.ericmedvet.jnb.core.Cacheable;
import io.github.ericmedvet.jnb.core.Discoverable;
import io.github.ericmedvet.jnb.core.Param;

@Discoverable(prefixTemplate = "ea.ttpn.gate")
public class NetworkGates {

  private NetworkGates() {
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate bAnd() {
    return Gates.bAnd();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate bConst(@Param("value") boolean value) {
    return Gates.bConst(value);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate bNot() {
    return Gates.bNot();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate bOr() {
    return Gates.bOr();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate bXor() {
    return Gates.bXor();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate concat() {
    return Gates.concat();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate equal() {
    return Gates.equal();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate iBefore() {
    return Gates.iBefore();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate iConst(@Param("value") int value) {
    return Gates.iConst(value);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate iPMathOperator(@Param("operator") Element.Operator operator) {
    return Gates.iPMathOperator(operator);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate iRange() {
    return Gates.iRange();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate iSMult() {
    return Gates.iSMult();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate iSPMult() {
    return Gates.iSPMult();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate iSPSum() {
    return Gates.iSPSum();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate iSSum() {
    return Gates.iSSum();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate iTh() {
    return Gates.iTh();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate iToR() {
    return Gates.iToR();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate length() {
    return Gates.length();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate noop() {
    return Gates.noop();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate pairer() {
    return Gates.pairer();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate queuer() {
    return Gates.queuer();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate rBefore() {
    return Gates.rBefore();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate rConst(@Param("value") double value) {
    return Gates.rConst(value);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate rPMathOperator(@Param("operator") Element.Operator operator) {
    return Gates.rPMathOperator(operator);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate rSMult() {
    return Gates.rSMult();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate rSPMult() {
    return Gates.rSPMult();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate rSPSum() {
    return Gates.rSPSum();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate rSSum() {
    return Gates.rSSum();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate rToI() {
    return Gates.rToI();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate repeater() {
    return Gates.repeater();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate sBefore() {
    return Gates.sBefore();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate sConcat() {
    return Gates.sConcat();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate sConst(@Param("value") String value) {
    return Gates.sConst(value);
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate sSplitter() {
    return Gates.sSplitter();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate select() {
    return Gates.select();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate sequencer() {
    return Gates.sequencer();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate sink() {
    return Gates.sink();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate splitter() {
    return Gates.splitter();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate unpairer() {
    return Gates.unpairer();
  }

  @SuppressWarnings("unused")
  @Cacheable
  public static Gate sPSequencer() {
    return Gates.sPSequencer();
  }
}
