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

package io.github.ericmedvet.jgea.core.representation.programsynthesis.ttpn;

import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Generic;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.Type;
import io.github.ericmedvet.jgea.core.representation.programsynthesis.type.TypeException;
import io.github.ericmedvet.jgea.core.util.Misc;
import io.github.ericmedvet.jgea.core.util.Sized;
import java.util.*;
import java.util.function.ToIntFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public final class Network implements Sized {
  private final SortedMap<Integer, Gate> gates;
  private final Set<Wire> wires;
  private final Map<Wire.EndPoint, Type> inputConcreteTypes;
  private final Map<Wire.EndPoint, Type> outputConcreteTypes;
  private final Map<Integer, Map<Generic, Type>> gateConcreteTypes;
  private final Map<Integer, Boolean> wiredToInputsMap;
  private final Map<Integer, Boolean> wiredToOutputsMap;
  private final Map<Integer, Boolean> deadMap;
  private List<Network> disjointSubnetwork;
  private List<List<Integer>> disjointSubnetworksGateIndexes;
  private Set<Wire.EndPoint> freeInputEndPoints;
  private Set<Wire.EndPoint> freeOutputEndPoints;

  public Network(List<Gate> gates, Set<Wire> wires) throws NetworkStructureException, TypeException {
    this(IntStream.range(0, gates.size()).boxed().collect(Misc.toSequencedMap(gates::get)), wires);
  }

  public Network(Map<Integer, Gate> gates, Set<Wire> wires) throws NetworkStructureException, TypeException {
    wiredToInputsMap = new HashMap<>();
    wiredToOutputsMap = new HashMap<>();
    deadMap = new HashMap<>();
    this.gates = Collections.unmodifiableSortedMap(new TreeMap<>(gates));
    this.wires = Collections.unmodifiableSortedSet(new TreeSet<>(wires));
    // validate wires
    for (Wire wire : wires) {
      try {
        validateGateIndexes(wire);
        validatePortIndexes(wire);
      } catch (NetworkStructureException e) {
        throw new NetworkStructureException("Wrong wire %s".formatted(wire), e);
      }
    }
    // validate gates
    for (int gateIndex : gates.keySet()) {
      try {
        validatePortArity(gateIndex);
        validateOutGenerics(gateIndex);
      } catch (NetworkStructureException e) {
        throw new NetworkStructureException("Wrong gate %d".formatted(gateIndex), e);
      }
    }
    // populate concrete types
    inputConcreteTypes = new LinkedHashMap<>();
    outputConcreteTypes = new LinkedHashMap<>();
    gateConcreteTypes = new LinkedHashMap<>();
    computeConcreteTypes();
    // validate types
    for (Wire wire : wires) {
      Type srcType = outputConcreteTypes.get(wire.src());
      Type dstType = inputConcreteTypes.get(wire.dst());
      if (srcType != null && dstType != null) {
        if (!dstType.canTakeValuesOf(srcType) && !srcType.canTakeValuesOf(dstType)) {
          throw new TypeException("Incompatible types on %s: %s on src, %s on dst".formatted(wire, srcType, dstType));
        }
      }
    }
  }

  public record Addition(Map<Integer, Gate> gates, Set<Wire> wires) {
    public static Addition empty() {
      return new Addition(Map.of(), Set.of());
    }

    public Network applyTo(Network network) throws NetworkStructureException, TypeException {
      if (isEmpty()) {
        return network;
      }
      if (network.gates.keySet().stream().anyMatch(gates::containsKey)) {
        throw new IllegalArgumentException(
            "Gates indexes already present: existing=%s, new=%s".formatted(network.gates.keySet(), gates.keySet())
        );
      }
      return new Network(
          Stream.concat(network.gates.entrySet().stream(), gates.entrySet().stream())
              .collect(
                  Collectors.toMap(
                      Map.Entry::getKey,
                      Map.Entry::getValue
                  )
              ),
          Stream.concat(network.wires.stream(), wires.stream()).collect(Collectors.toCollection(LinkedHashSet::new))
      );
    }

    public boolean isEmpty() {
      return gates.isEmpty() && wires.isEmpty();
    }
  }

  public record Deletion(List<Integer> gateIndexes, Set<Wire> wires) {
    public static Deletion empty() {
      return new Deletion(List.of(), Set.of());
    }

    public Network applyTo(Network network) throws NetworkStructureException, TypeException {
      if (isEmpty()) {
        return network;
      }
      Map<Integer, Gate> newGates = new HashMap<>(network.gates());
      gateIndexes.forEach(newGates::remove);
      SequencedSet<Wire> newWires = new LinkedHashSet<>(network.wires());
      wires.forEach(newWires::remove);
      return new Network(newGates, newWires);
    }

    public boolean isEmpty() {
      return gateIndexes.isEmpty() && wires.isEmpty();
    }
  }

  private record TypedEndPoint(Wire.EndPoint endPoint, Type type) {}

  private static boolean replaceType(Type type, Wire.EndPoint endPoint, Map<Wire.EndPoint, Type> map) {
    Type existingType = map.get(endPoint);
    if (!type.canTakeValuesOf(existingType)) {
      map.put(endPoint, type);
      return true;
    }
    return false;
  }

  private void checkGenericAssignments(Set<Type> types) throws TypeException {
    List<Type> concreteTypes = types.stream().filter(t -> !t.isGeneric()).toList();
    List<Type> genericTypes = types.stream().filter(Type::isGeneric).toList();
    if (concreteTypes.size() > 1) {
      throw new TypeException("Multiple concrete values: %s".formatted(concreteTypes));
    }
    if (!concreteTypes.isEmpty()) {
      Type concreteType = concreteTypes.getFirst();
      for (Type genericType : genericTypes) {
        if (!genericType.canTakeValuesOf(concreteType)) {
          throw new TypeException("Inconsistent generic value: %s cannot take %s".formatted(genericType, concreteType));
        }
      }
    } else {
      for (int i = 0; i < genericTypes.size(); i++) {
        for (int j = i + 1; j < genericTypes.size(); i++) {
          Type genericType1 = genericTypes.get(i);
          Type genericType2 = genericTypes.get(j);
          if (genericType1.canTakeValuesOf(genericType2) && genericType2.canTakeValuesOf(genericType1)) {
            throw new TypeException("Inconsistent generic values: %s and %s".formatted(genericType1, genericType2));
          }
        }
      }
    }
  }

  private void computeConcreteTypes() throws TypeException {
    // fill with non-generic types
    for (int gi : gates.keySet()) {
      Gate gate = gates.get(gi);
      for (int pi = 0; pi < gate.inputPorts().size(); pi = pi + 1) {
        if (!gate.inputPorts().get(pi).type().isGeneric()) {
          inputConcreteTypes.put(new Wire.EndPoint(gi, pi), gate.inputPorts().get(pi).type());
        }
      }
      for (int pi = 0; pi < gate.outputTypes().size(); pi = pi + 1) {
        if (!gate.outputTypes().get(pi).isGeneric()) {
          outputConcreteTypes.put(new Wire.EndPoint(gi, pi), gate.outputTypes().get(pi));
        }
      }
    }
    // iteratively propagate types
    while (true) {
      boolean changed = false;
      // propagate through wires
      for (Wire wire : wires) {
        Type srcType = outputConcreteTypes.get(wire.src());
        Type dstType = inputConcreteTypes.get(wire.dst());
        if (srcType == null && dstType != null) {
          Type pType = outputConcreteTypes.put(wire.src(), dstType);
          changed = changed || pType == null;
        }
        if (srcType != null && dstType == null) {
          Type pType = inputConcreteTypes.put(wire.dst(), srcType);
          changed = changed || pType == null;
        }
      }
      // compute gate concrete types
      boolean updatedMap = updateGateConcreteMaps();
      changed = changed || updatedMap;
      // propagate through gate
      for (Map.Entry<Integer, Map<Generic, Type>> entry : gateConcreteTypes.entrySet()) {
        int gi = entry.getKey();
        Gate gate = gates.get(gi);
        for (int pi = 0; pi < gate.inputPorts().size(); pi = pi + 1) {
          boolean localChanged = replaceType(
              gate.inputPorts().get(pi).type().concrete(entry.getValue()),
              new Wire.EndPoint(gi, pi),
              inputConcreteTypes
          );
          changed = localChanged || changed;
        }
        for (int pi = 0; pi < gate.outputTypes().size(); pi = pi + 1) {
          boolean localChanged = replaceType(
              gate.outputTypes().get(pi).concrete(entry.getValue()),
              new Wire.EndPoint(gi, pi),
              outputConcreteTypes
          );
          changed = localChanged || changed;
        }
      }
      if (!changed) {
        break;
      }
    }
  }

  public Type concreteInputType(Wire.EndPoint endPoint) {
    return inputConcreteTypes.getOrDefault(
        endPoint,
        gates.get(endPoint.gateIndex()).inputPorts().get(endPoint.portIndex()).type()
    );
  }

  public Map<Generic, Type> concreteMapping(int gi) {
    return gateConcreteTypes.getOrDefault(gi, Map.of());
  }

  public Type concreteOutputType(Wire.EndPoint endPoint) {
    return outputConcreteTypes.getOrDefault(
        endPoint,
        gates.get(endPoint.gateIndex()).outputTypes().get(endPoint.portIndex())
    );
  }

  public List<Integer> deadGates() {
    return gates.keySet().stream().filter(this::isDeadGate).toList();
  }

  public List<Integer> deadOrIUnwiredGates() {
    return gates.keySet().stream().filter(gi -> isDeadGate(gi) || !isWiredToInput(gi)).toList();
  }

  public SortedMap<Integer, Type> deadOrIUnwiredOutputGates() {
    return new TreeMap<>(
        outputGates().entrySet()
            .stream()
            .filter(e -> isDeadGate(e.getKey()) || !isWiredToInput(e.getKey()))
            .collect(
                Collectors.toMap(
                    Map.Entry::getKey,
                    Map.Entry::getValue
                )
            )
    );
  }

  public SortedMap<Integer, Type> deadOutputGates() {
    return new TreeMap<>(
        outputGates().entrySet()
            .stream()
            .filter(e -> isDeadGate(e.getKey()))
            .collect(
                Collectors.toMap(
                    Map.Entry::getKey,
                    Map.Entry::getValue
                )
            )
    );
  }

  public synchronized List<Network> disjointSubnetworks() throws NetworkStructureException, TypeException {
    if (disjointSubnetwork != null) {
      return disjointSubnetwork;
    }
    disjointSubnetwork = new ArrayList<>();
    for (List<Integer> subnetIndexes : disjointSubnetworksGateIndexes()) {
      Network network = new Network(
          subnetIndexes.stream().collect(Collectors.toMap(gi -> gi, gates::get)),
          wires()
              .stream()
              .filter(w -> subnetIndexes.contains(w.src().gateIndex()) && subnetIndexes.contains(w.dst().gateIndex()))
              .collect(Collectors.toCollection(LinkedHashSet::new))
      );
      disjointSubnetwork.add(network);
    }
    return disjointSubnetwork;
  }

  public synchronized List<List<Integer>> disjointSubnetworksGateIndexes() {
    if (disjointSubnetworksGateIndexes != null) {
      return disjointSubnetworksGateIndexes;
    }
    disjointSubnetworksGateIndexes = new ArrayList<>();
    SequencedSet<Integer> allGis = new LinkedHashSet<>(gates.keySet());
    while (!allGis.isEmpty()) {
      SequencedSet<Integer> subnetGis = new LinkedHashSet<>();
      findWiredGates(allGis.getFirst(), subnetGis);
      disjointSubnetworksGateIndexes.add(subnetGis.stream().toList());
      allGis.removeAll(subnetGis);
    }
    return disjointSubnetworksGateIndexes;
  }

  private void findWiredGates(int gi, SequencedSet<Integer> gis) {
    if (gis.contains(gi)) {
      return;
    }
    gis.add(gi);
    for (Wire w : wiresFrom(gi)) {
      findWiredGates(w.dst().gateIndex(), gis);
    }
    for (Wire w : wiresTo(gi)) {
      findWiredGates(w.src().gateIndex(), gis);
    }
  }

  public synchronized Set<Wire.EndPoint> freeInputEndPoints() {
    if (freeInputEndPoints != null) {
      return freeInputEndPoints;
    }
    freeInputEndPoints = gates.keySet()
        .stream()
        .flatMap(
            gi -> IntStream.range(0, gates.get(gi).inputPorts().size())
                .mapToObj(pi -> new Wire.EndPoint(gi, pi))
        )
        .filter(ep -> wireTo(ep).isEmpty())
        .collect(Collectors.toCollection(LinkedHashSet::new));
    return freeInputEndPoints;
  }

  public Set<Wire.EndPoint> freeInputEndPoints(int gi) {
    return IntStream.range(0, gates.get(gi).inputPorts().size())
        .mapToObj(pi -> new Wire.EndPoint(gi, pi))
        .filter(ep -> wireTo(ep).isEmpty())
        .collect(Collectors.toCollection(LinkedHashSet::new));
  }

  public Set<Wire.EndPoint> freeOutputEndPoints(int gi) {
    return IntStream.range(0, gates.get(gi).outputTypes().size())
        .mapToObj(pi -> new Wire.EndPoint(gi, pi))
        .filter(ep -> wiresFrom(ep).isEmpty())
        .collect(Collectors.toCollection(LinkedHashSet::new));
  }

  public synchronized Set<Wire.EndPoint> freeOutputEndPoints() {
    if (freeOutputEndPoints != null) {
      return freeOutputEndPoints;
    }
    freeOutputEndPoints = gates.keySet()
        .stream()
        .flatMap(
            gi -> IntStream.range(0, gates.get(gi).outputTypes().size())
                .mapToObj(pi -> new Wire.EndPoint(gi, pi))
        )
        .filter(ep -> wiresFrom(ep).isEmpty())
        .collect(Collectors.toCollection(LinkedHashSet::new));
    return freeOutputEndPoints;
  }

  public SortedMap<Integer, Gate> gates() {
    return gates;
  }

  @Override
  public int hashCode() {
    return Objects.hash(gates, wires);
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == this)
      return true;
    if (obj == null || obj.getClass() != this.getClass())
      return false;
    var that = (Network) obj;
    return Objects.equals(this.gates, that.gates) && Objects.equals(this.wires, that.wires);
  }

  @Override
  public String toString() {
    return gates.entrySet()
        .stream()
        .map(
            e -> "%3d%1.1s: %s %s(%s)-->(%s)"
                .formatted(
                    e.getKey(),
                    isDeadGate(e.getKey()) ? "*" : "",
                    e.getValue(),
                    (e.getValue().hasGenerics() && !gateConcreteTypes.get(e.getKey()).isEmpty()) ? "{with %s} "
                        .formatted(
                            gateConcreteTypes.get(e.getKey())
                                .entrySet()
                                .stream()
                                .map(Object::toString)
                                .collect(Collectors.joining(","))
                        ) : "",
                    IntStream.range(0, e.getValue().inputPorts().size())
                        .mapToObj(
                            j -> wireTo(new Wire.EndPoint(e.getKey(), j))
                                .map(w -> w.src().toString())
                                .orElse("_")
                        )
                        .collect(Collectors.joining(",")),
                    IntStream.range(0, e.getValue().outputTypes().size())
                        .mapToObj(
                            j -> (wiresFrom(new Wire.EndPoint(e.getKey(), j)).isEmpty() ? "_" : wiresFrom(
                                new Wire.EndPoint(
                                    e.getKey(),
                                    j
                                )
                            ).stream()
                                .map(w -> w.dst().toString())
                                .collect(Collectors.joining("+")))
                        )
                        .collect(Collectors.joining(","))
                )
        )
        .collect(Collectors.joining("\n"));
  }

  public int inputDistanceFrom(Class<? extends Gate> gateClass, int gi) {
    return inputDistanceFrom(gateClass, gi, new HashSet<>());
  }

  private int inputDistanceFrom(Class<? extends Gate> gateClass, int gi, Set<Integer> visitedGis) {
    visitedGis.add(gi);
    if (gateClass.isAssignableFrom(gates().get(gi).getClass())) {
      return 0;
    }
    Set<Wire> wires = wiresTo(gi);
    if (wires.isEmpty()) {
      return Integer.MAX_VALUE;
    }
    int d = wires.stream()
        .filter(w -> !visitedGis.contains(w.src().gateIndex()))
        .mapToInt(w -> inputDistanceFrom(gateClass, w.src().gateIndex(), visitedGis))
        .min()
        .orElse(Integer.MAX_VALUE);
    return (d == Integer.MAX_VALUE) ? Integer.MAX_VALUE : (d + 1);
  }

  public SortedMap<Integer, Type> inputGates() {
    return new TreeMap<>(
        gates.entrySet()
            .stream()
            .filter(e -> e.getValue() instanceof Gate.InputGate)
            .collect(
                Collectors.toMap(
                    Map.Entry::getKey,
                    e -> ((Gate.InputGate) e.getValue()).type()
                )
            )
    );
  }

  public Type inputType(Wire.EndPoint endPoint) {
    return gates.get(endPoint.gateIndex()).inputPorts().get(endPoint.portIndex()).type();
  }

  public synchronized boolean isDeadGate(int igi) {
    Boolean b = deadMap.get(igi);
    if (b != null) {
      return b;
    }
    Gate gate = gates.get(igi);
    for (int pi = 0; pi < gate.inputPorts().size(); pi = pi + 1) {
      if (gate.inputPorts().get(pi).n() > 0) {
        if (wireTo(igi, pi).isEmpty()) {
          deadMap.put(igi, true);
          return true;
        }
        Set<Wire.EndPoint> endPoints = new HashSet<>();
        Set<Integer> gateIndexes = new HashSet<>();
        wiredInputEndPoints(new Wire.EndPoint(igi, pi), gateIndexes, endPoints);
        if (gateIndexes.contains(igi)) {
          deadMap.put(igi, true);
          return true;
        }
        if (gateIndexes.stream().anyMatch(this::isDeadGate)) {
          deadMap.put(igi, true);
          return true;
        }
      }
    }
    deadMap.put(igi, false);
    return false;
  }

  public synchronized boolean isWiredToInput(int gi) {
    return wiredToInputsMap.computeIfAbsent(
        gi,
        igi -> {
          Set<Wire.EndPoint> endPoints = new HashSet<>();
          Set<Integer> gateIndexes = new HashSet<>();
          wiredInputEndPoints(igi, gateIndexes, endPoints);
          return gateIndexes.stream().anyMatch(localGi -> gates.get(localGi) instanceof Gate.InputGate);
        }
    );
  }

  public synchronized boolean isWiredToOutput(int gi) {
    return wiredToOutputsMap.computeIfAbsent(
        gi,
        igi -> {
          Set<Wire.EndPoint> endPoints = new HashSet<>();
          Set<Integer> gateIndexes = new HashSet<>();
          wiredOutputEndPoints(igi, gateIndexes, endPoints);
          return gateIndexes.stream().anyMatch(localGi -> gates.get(localGi) instanceof Gate.OutputGate);
        }
    );
  }

  public int outputDistanceFrom(Class<? extends Gate> gateClass, int gi) {
    return outputDistanceFrom(gateClass, gi, new HashSet<>());
  }

  private int outputDistanceFrom(Class<? extends Gate> gateClass, int gi, Set<Integer> visitedGis) {
    visitedGis.add(gi);
    if (gateClass.isAssignableFrom(gates().get(gi).getClass())) {
      return 0;
    }
    Set<Wire> wires = wiresFrom(gi);
    if (wires.isEmpty()) {
      return Integer.MAX_VALUE;
    }
    int d = wires.stream()
        .filter(w -> !visitedGis.contains(w.dst().gateIndex()))
        .mapToInt(w -> outputDistanceFrom(gateClass, w.dst().gateIndex(), visitedGis))
        .min()
        .orElse(Integer.MAX_VALUE);
    return (d == Integer.MAX_VALUE) ? Integer.MAX_VALUE : (d + 1);
  }

  public SortedMap<Integer, Type> outputGates() {
    return new TreeMap<>(
        gates.entrySet()
            .stream()
            .filter(e -> e.getValue() instanceof Gate.OutputGate)
            .collect(
                Collectors.toMap(
                    Map.Entry::getKey,
                    e -> ((Gate.OutputGate) e.getValue()).type()
                )
            )
    );
  }

  public Set<Wire.EndPoint> outputPorts() {
    return gates.entrySet()
        .stream()
        .flatMap(
            e -> IntStream.range(0, e.getValue().outputTypes().size())
                .mapToObj(pi -> new Wire.EndPoint(e.getKey(), pi))
        )
        .collect(Collectors.toCollection(LinkedHashSet::new));
  }

  public Type outputType(Wire.EndPoint endPoint) {
    return gates.get(endPoint.gateIndex()).outputTypes().get(endPoint.portIndex());
  }

  @Override
  public int size() {
    return gates().size() + wires.size();
  }

  private boolean updateGateConcreteMaps() throws TypeException {
    int initialMapSize = gateConcreteTypes.size();
    for (int gi : gates.keySet()) {
      Gate gate = gates.get(gi);
      if (gate.hasGenerics()) {
        List<Map<Generic, Type>> maps = new ArrayList<>();
        // add from input ports
        for (int j = 0; j < gate.inputPorts().size(); j++) {
          Optional<Wire> oToWire = wireTo(new Wire.EndPoint(gi, j));
          if (oToWire.isPresent()) {
            Wire toWire = oToWire.get();
            Type concreteType = concreteOutputType(toWire.src());
            if (!concreteType.isGeneric()) {
              maps.add(gate.inputPorts().get(j).type().resolveGenerics(concreteType));
            }
          }
        }
        // add from output ports
        for (int j = 0; j < gate.outputTypes().size(); j++) {
          for (Wire fromWire : wiresFrom(new Wire.EndPoint(gi, j))) {
            Type concreteType = concreteInputType(fromWire.dst());
            if (!concreteType.isGeneric()) {
              maps.add(gate.outputTypes().get(j).resolveGenerics(concreteType));
            }
          }
        }
        // merge and check
        Map<Generic, Set<Type>> merged = Misc.merge(maps);
        for (Map.Entry<Generic, Set<Type>> entry : merged.entrySet()) {
          try {
            checkGenericAssignments(entry.getValue());
          } catch (TypeException e) {
            throw new TypeException("Multiple assignments types for %s".formatted(entry.getKey()), e);
          }
        }
        // map generic to actual types
        Map<Generic, Type> map = merged.entrySet()
            .stream()
            .collect(
                Collectors.toMap(
                    Map.Entry::getKey,
                    e -> e.getValue()
                        .stream()
                        .reduce((t1, t2) -> t1.canTakeValuesOf(t2) ? t2 : t1)
                        .orElseThrow()
                )
            );
        gateConcreteTypes.put(gi, map);
      }
    }
    return initialMapSize != gateConcreteTypes.size();
  }

  private void validateGateIndexes(Wire wire) throws NetworkStructureException {
    if (!gates.containsKey(wire.src().gateIndex())) {
      throw new NetworkStructureException("Not existing src gate");
    }
    if (!gates.containsKey(wire.dst().gateIndex())) {
      throw new NetworkStructureException("Not existing dst gate");
    }
  }

  private void validateOutGenerics(int gateIndex) throws NetworkStructureException {
    Set<Generic> outGenerics = gates.get(gateIndex)
        .outputTypes()
        .stream()
        .map(Type::generics)
        .reduce(Misc::union)
        .orElse(Set.of());
    Set<Generic> inGenerics = gates.get(gateIndex)
        .inputPorts()
        .stream()
        .map(Gate.Port::type)
        .map(Type::generics)
        .reduce(Misc::union)
        .orElse(Set.of());
    if (!inGenerics.containsAll(outGenerics)) {
      throw new NetworkStructureException(
          "Undefined out generics: in=%s out=%s".formatted(inGenerics, outGenerics)
      );
    }
  }

  private void validatePortArity(int gateIndex) throws NetworkStructureException {
    for (int portIndex = 0; portIndex < gates.get(gateIndex).inputPorts().size(); portIndex++) {
      int finalPortIndex = portIndex;
      long count = wires.stream()
          .filter(w -> w.dst().gateIndex() == gateIndex && w.dst().portIndex() == finalPortIndex)
          .count();
      if (count > 1) {
        throw new NetworkStructureException("Multiple input wires on port %d".formatted(portIndex));
      }
    }
  }

  private void validatePortIndexes(Wire wire) throws NetworkStructureException {
    if (gates.get(wire.src().gateIndex()).outputTypes().size() <= wire.src().portIndex()) {
      throw new NetworkStructureException(
          "Not existing out port in src gate (%s)"
              .formatted(gates.get(wire.src().gateIndex()))
      );
    }
    if (gates.get(wire.dst().gateIndex()).inputPorts().size() <= wire.dst().portIndex()) {
      throw new NetworkStructureException(
          "Not existing in port in dst gate (%s)"
              .formatted(gates.get(wire.dst().gateIndex()))
      );
    }
  }

  public Network wireFreeInputEndPoints(
      ToIntFunction<List<Type>> chooser
  ) throws NetworkStructureException, TypeException {
    List<TypedEndPoint> iTEPs = new ArrayList<>(
        freeInputEndPoints().stream()
            .map(ep -> new TypedEndPoint(ep, inputType(ep)))
            .toList()
    );
    while (true) {
      if (iTEPs.isEmpty()) {
        return this;
      }
      int iIndex = chooser.applyAsInt(iTEPs.stream().map(TypedEndPoint::type).toList());
      TypedEndPoint iTEP = iTEPs.get(iIndex);
      List<TypedEndPoint> oTEPs = new ArrayList<>(
          freeOutputEndPoints().stream()
              .map(ep -> new TypedEndPoint(ep, concreteOutputType(ep)))
              .filter(oTEP -> iTEP.type.canTakeValuesOf(oTEP.type))
              .toList()
      );
      if (oTEPs.isEmpty()) {
        oTEPs = new ArrayList<>(
            outputPorts().stream()
                .map(ep -> new TypedEndPoint(ep, concreteOutputType(ep)))
                .filter(oTEP -> iTEP.type.canTakeValuesOf(oTEP.type))
                .toList()
        );
      }
      while (true) {
        if (oTEPs.isEmpty()) {
          iTEPs.remove(iIndex);
          break;
        }
        int oIndex = chooser.applyAsInt(oTEPs.stream().map(TypedEndPoint::type).toList());
        Set<Wire> newWires = new LinkedHashSet<>(wires());
        newWires.add(new Wire(oTEPs.get(oIndex).endPoint, iTEP.endPoint));
        try {
          return new Network(gates, newWires).wireFreeInputEndPoints(chooser);
        } catch (TypeException e) {
          oTEPs.remove(oIndex);
        }
      }
    }
  }

  public Network wireFreeOutputEndPoints(
      ToIntFunction<List<Type>> chooser
  ) throws NetworkStructureException, TypeException {
    List<TypedEndPoint> oTEPs = new ArrayList<>(
        freeOutputEndPoints().stream()
            .map(ep -> new TypedEndPoint(ep, outputType(ep)))
            .toList()
    );
    while (true) {
      if (oTEPs.isEmpty()) {
        return this;
      }
      int oIndex = chooser.applyAsInt(oTEPs.stream().map(TypedEndPoint::type).toList());
      TypedEndPoint oTEP = oTEPs.get(oIndex);
      List<TypedEndPoint> iTEPs = new ArrayList<>(
          freeInputEndPoints().stream()
              .map(ep -> new TypedEndPoint(ep, inputType(ep)))
              .filter(iTEP -> iTEP.type.canTakeValuesOf(oTEP.type))
              .toList()
      );
      while (true) {
        if (iTEPs.isEmpty()) {
          oTEPs.remove(oIndex);
          break;
        }
        int iIndex = chooser.applyAsInt(iTEPs.stream().map(TypedEndPoint::type).toList());
        Set<Wire> newWires = new LinkedHashSet<>(wires());
        newWires.add(new Wire(oTEP.endPoint, iTEPs.get(iIndex).endPoint));
        try {
          return new Network(gates, newWires).wireFreeOutputEndPoints(chooser);
        } catch (TypeException e) {
          iTEPs.remove(iIndex);
        }
      }
    }
  }

  private Optional<Wire> wireTo(Wire.EndPoint dst) {
    return wires.stream().filter(w -> w.dst().equals(dst)).findFirst();
  }

  public Optional<Wire> wireTo(int gateIndex, int portIndex) {
    return wireTo(new Wire.EndPoint(gateIndex, portIndex));
  }

  private void wiredInputEndPoints(Wire.EndPoint endPoint, Set<Integer> gateIndexes, Set<Wire.EndPoint> endPoints) {
    if (endPoints.contains(endPoint)) {
      return;
    }
    endPoints.add(endPoint);
    wireTo(endPoint).ifPresent(w -> wiredInputEndPoints(w.src().gateIndex(), gateIndexes, endPoints));
  }

  private void wiredInputEndPoints(int gi, Set<Integer> gateIndexes, Set<Wire.EndPoint> endPoints) {
    gateIndexes.add(gi);
    IntStream.range(0, gates.get(gi).inputPorts().size())
        .forEach(
            pi -> wiredInputEndPoints(
                new Wire.EndPoint(gi, pi),
                gateIndexes,
                endPoints
            )
        );
  }

  private void wiredOutputEndPoints(Wire.EndPoint endPoint, Set<Integer> gateIndexes, Set<Wire.EndPoint> endPoints) {
    if (endPoints.contains(endPoint)) {
      return;
    }
    endPoints.add(endPoint);
    wiresFrom(endPoint).forEach(w -> wiredOutputEndPoints(w.dst().gateIndex(), gateIndexes, endPoints));
  }

  private void wiredOutputEndPoints(int gi, Set<Integer> gateIndexes, Set<Wire.EndPoint> endPoints) {
    gateIndexes.add(gi);
    IntStream.range(0, gates.get(gi).outputTypes().size())
        .forEach(
            pi -> wiredOutputEndPoints(
                new Wire.EndPoint(gi, pi),
                gateIndexes,
                endPoints
            )
        );
  }

  public Set<Wire> wires() {
    return wires;
  }

  public Set<Wire> wiresFrom(int gateIndex) {
    return IntStream.range(0, gates.get(gateIndex).outputTypes().size())
        .mapToObj(pi -> wiresFrom(gateIndex, pi))
        .flatMap(List::stream)
        .collect(Collectors.toCollection(LinkedHashSet::new));
  }

  private List<Wire> wiresFrom(Wire.EndPoint src) {
    return wires.stream().filter(w -> w.src().equals(src)).toList();
  }

  public List<Wire> wiresFrom(int gateIndex, int portIndex) {
    return wiresFrom(new Wire.EndPoint(gateIndex, portIndex));
  }

  public Set<Wire> wiresTo(int gateIndex) {
    return IntStream.range(0, gates.get(gateIndex).inputPorts().size())
        .mapToObj(pi -> wireTo(gateIndex, pi))
        .filter(Optional::isPresent)
        .map(Optional::get)
        .collect(Collectors.toCollection(LinkedHashSet::new));
  }

}
