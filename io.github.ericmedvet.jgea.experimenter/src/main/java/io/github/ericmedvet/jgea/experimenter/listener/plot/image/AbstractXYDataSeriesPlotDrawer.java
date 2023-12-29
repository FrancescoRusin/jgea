/*-
 * ========================LICENSE_START=================================
 * jgea-experimenter
 * %%
 * Copyright (C) 2018 - 2023 Eric Medvet
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
package io.github.ericmedvet.jgea.experimenter.listener.plot.image;

import io.github.ericmedvet.jgea.experimenter.listener.plot.XYDataSeries;
import io.github.ericmedvet.jgea.experimenter.listener.plot.XYDataSeriesPlot;
import io.github.ericmedvet.jgea.experimenter.listener.plot.XYPlot;
import io.github.ericmedvet.jsdynsym.core.DoubleRange;
import io.github.ericmedvet.jsdynsym.grid.Grid;
import java.awt.*;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.awt.geom.Rectangle2D;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SortedMap;

public abstract class AbstractXYDataSeriesPlotDrawer extends AbstractPlotDrawer<XYDataSeriesPlot, List<XYDataSeries>> {

  protected final SortedMap<String, Color> dataColors;

  public AbstractXYDataSeriesPlotDrawer(ImagePlotter ip, XYDataSeriesPlot plot, List<Color> colors) {
    super(ip, plot);
    dataColors = ip.computeSeriesDataColors(
        plot.dataGrid().values().stream()
            .filter(Objects::nonNull)
            .map(XYPlot.TitledData::data)
            .flatMap(List::stream)
            .distinct()
            .toList(),
        colors);
  }

  @Override
  public double computeNoteH(Graphics2D g, Grid.Key k) {
    return 0;
  }

  @Override
  public void drawNote(Graphics2D g, Rectangle2D r, Grid.Key k) {}

  protected abstract Point2D computeLegendImageSize();

  protected abstract void drawData(Graphics2D g, Rectangle2D r, Axis xA, Axis yA, XYDataSeries ds, Color color);

  protected abstract void drawLegendImage(Graphics2D g, Rectangle2D r, Color color);

  @Override
  public double computeLegendH(Graphics2D g) {
    double maxLineL = ip.w() - 2d * ip.c().layout().legendMarginWRate() * ip.w();
    double lineH = Math.max(
        computeLegendImageSize().getY(), ip.computeStringH(g, "0", Configuration.Text.Use.LEGEND_LABEL));
    double lH = lineH;
    double lineL = 0;
    for (String s : dataColors.keySet()) {
      double localL = computeLegendImageSize().getX()
          + 2d * ip.c().layout().legendInnerMarginWRate() * ip.w()
          + ip.computeStringW(g, s, Configuration.Text.Use.LEGEND_LABEL);
      if (lineL + localL > maxLineL) {
        lH = lH + ip.c().layout().legendInnerMarginHRate() * ip.h() + lineH;
        lineL = 0;
      }
      lineL = lineL + localL;
    }
    return lH;
  }

  @Override
  protected DoubleRange computeRange(List<XYDataSeries> data, boolean isXAxis) {
    return data.stream()
        .map(d -> isXAxis ? d.xRange() : d.yRange())
        .reduce((r1, r2) -> new DoubleRange(Math.min(r1.min(), r2.min()), Math.max(r1.max(), r2.max())))
        .orElseThrow();
  }

  @Override
  public void drawLegend(Graphics2D g, Rectangle2D r) {
    if (ip.c().debug()) {
      g.setStroke(new BasicStroke(1));
      g.setColor(Color.MAGENTA);
      g.draw(r);
    }
    double lineH = Math.max(
        computeLegendImageSize().getY(), ip.computeStringH(g, "0", Configuration.Text.Use.LEGEND_LABEL));
    double x = 0;
    double y = 0;
    for (Map.Entry<String, Color> e : dataColors.entrySet()) {
      double localL = computeLegendImageSize().getX()
          + 2d * ip.c().layout().legendInnerMarginWRate() * ip.w()
          + ip.computeStringW(g, e.getKey(), Configuration.Text.Use.LEGEND_LABEL);
      if (x + localL > r.getWidth()) {
        y = y + ip.c().layout().legendInnerMarginHRate() * ip.h() + lineH;
        x = 0;
      }
      Rectangle2D legendImageR = new Rectangle2D.Double(
          r.getX() + x,
          r.getY() + y,
          computeLegendImageSize().getX(),
          computeLegendImageSize().getY());
      g.setColor(ip.c().colors().plotBgColor());
      g.fill(legendImageR);
      drawLegendImage(g, legendImageR, e.getValue());
      ip.drawString(
          g,
          new Point2D.Double(
              r.getX()
                  + x
                  + legendImageR.getWidth()
                  + ip.c().layout().legendInnerMarginWRate() * ip.w(),
              r.getY() + y),
          e.getKey(),
          ImagePlotter.AnchorH.L,
          ImagePlotter.AnchorV.B,
          Configuration.Text.Use.LEGEND_LABEL,
          Configuration.Text.Direction.H,
          ip.c().colors().titleColor());
      x = x + localL;
    }
  }

  @Override
  public void drawPlot(Graphics2D g, Rectangle2D r, Grid.Key k, Axis xA, Axis yA) {
    g.setColor(ip.c().colors().gridColor());
    xA.ticks()
        .forEach(x -> g.draw(new Line2D.Double(
            xA.xIn(x, r), yA.yIn(yA.range().min(), r),
            xA.xIn(x, r), yA.yIn(yA.range().max(), r))));
    yA.ticks()
        .forEach(y -> g.draw(new Line2D.Double(
            xA.xIn(xA.range().min(), r), yA.yIn(y, r),
            xA.xIn(xA.range().max(), r), yA.yIn(y, r))));
    if (ip.c().debug()) {
      g.setStroke(new BasicStroke(1));
      g.setColor(Color.MAGENTA);
      g.draw(r);
    }
    // draw data
    plot.dataGrid().get(k).data().forEach(ds -> drawData(g, r, xA, yA, ds, dataColors.get(ds.name())));
  }
}
