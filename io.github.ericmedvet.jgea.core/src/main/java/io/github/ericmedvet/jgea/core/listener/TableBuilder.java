package io.github.ericmedvet.jgea.core.listener;

import io.github.ericmedvet.jgea.core.util.ArrayTable;
import io.github.ericmedvet.jgea.core.util.Misc;
import io.github.ericmedvet.jgea.core.util.Table;

import java.util.List;
public class TableBuilder<E, O, K> implements AccumulatorFactory<E, Table<O>, K> {

  private final List<NamedFunction<? super E, ? extends O>> eFunctions;
  private final List<NamedFunction<? super K, ? extends O>> kFunctions;

  public TableBuilder(
      List<NamedFunction<? super E, ? extends O>> eFunctions, List<NamedFunction<? super K, ? extends O>> kFunctions
  ) {
    this.eFunctions = eFunctions;
    this.kFunctions = kFunctions;
  }

  @Override
  public Accumulator<E, Table<O>> build(K k) {
    List<? extends O> kValues = kFunctions.stream().map(f -> f.apply(k)).toList();
    return new Accumulator<>() {

      private final Table<O> table = new ArrayTable<>(Misc.concat(List.of(kFunctions, eFunctions))
          .stream()
          .map(NamedFunction::getName)
          .toList());

      @Override
      public Table<O> get() {
        return table;
      }

      @Override
      public void listen(E e) {
        List<? extends O> eValues = eFunctions.stream().map(f -> f.apply(e)).toList();
        table.addRow(Misc.concat(List.of(kValues, eValues)));
      }
    };
  }

}
