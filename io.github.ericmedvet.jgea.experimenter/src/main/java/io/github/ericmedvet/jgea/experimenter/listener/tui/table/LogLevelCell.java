package io.github.ericmedvet.jgea.experimenter.listener.tui.table;

import com.googlecode.lanterna.TextColor;
import io.github.ericmedvet.jgea.experimenter.listener.tui.util.TuiDrawer;

import java.util.Map;
import java.util.logging.Level;

/**
 * @author "Eric Medvet" on 2023/11/10 for jgea
 */
public record LogLevelCell(Level level) implements Cell {
  private static final String LEVEL_FORMAT = "%4.4s";
  private static final Map<Level, TextColor> LEVEL_COLORS =
      Map.ofEntries(
          Map.entry(Level.SEVERE, TextColor.Factory.fromString("#EE3E38")),
          Map.entry(Level.WARNING, TextColor.Factory.fromString("#FBA465")),
          Map.entry(Level.INFO, TextColor.Factory.fromString("#D8E46B")),
          Map.entry(Level.CONFIG, TextColor.Factory.fromString("#6D8700"))
      );

  @Override
  public void draw(TuiDrawer td, int width) {
    td.drawString(0, 0, LEVEL_FORMAT.formatted(level.toString()), LEVEL_COLORS.getOrDefault(level, td.getConfiguration()
        .primaryStringColor()));
  }

  @Override
  public int preferredWidth() {
    return LEVEL_FORMAT.formatted(level.toString()).length();
  }
}
