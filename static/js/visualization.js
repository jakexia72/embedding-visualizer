/**
 * Custom D3.js visualization for embeddings
 */

class EmbeddingsVisualizer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.width = this.container.clientWidth;
    this.height = 600; // Fixed height for consistency
    this.margin = { top: 20, right: 20, bottom: 20, left: 20 };
    this.data = null;
    this.svg = null;
    this.tooltip = null;
    this.zoom = null;
  }

  // Handle zoom behavior
  handleZoom(event) {
    const { transform } = event;
    this.svg.select(".plot-area").attr("transform", transform);
  }

  // Initialize the visualization with data
  initialize(data) {
    this.data = data;
    this.container.innerHTML = "";
    
    // Create tooltip
    if (document.querySelector('.vis-tooltip')) {
      document.querySelector('.vis-tooltip').remove();
    }
    
    this.tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "vis-tooltip")
      .style("opacity", 0);
    
    // Initialize 2D visualization
    this.initialize2D();
  }

  // Initialize 2D visualization
  initialize2D() {
    // Set up SVG
    this.svg = d3
      .select(this.container)
      .append("svg")
      .attr("width", this.width)
      .attr("height", this.height)
      .attr("class", "visualization-svg");

    // Set up zoom behavior
    this.zoom = d3
      .zoom()
      .scaleExtent([0.1, 10])
      .on("zoom", (event) => this.handleZoom(event));

    this.svg.call(this.zoom);

    // Create a group for the plot area
    const plotArea = this.svg
      .append("g")
      .attr("class", "plot-area")
      .attr(
        "transform",
        `translate(${this.margin.left}, ${this.margin.top})`
      );

    // Calculate scales
    const xExtent = d3.extent(this.data.points, (d) => d.x);
    const yExtent = d3.extent(this.data.points, (d) => d.y);

    // Add padding to the extents
    const xPadding = (xExtent[1] - xExtent[0]) * 0.05;
    const yPadding = (yExtent[1] - yExtent[0]) * 0.05;

    const xScale = d3
      .scaleLinear()
      .domain([xExtent[0] - xPadding, xExtent[1] + xPadding])
      .range([this.margin.left, this.width - this.margin.right]);

    const yScale = d3
      .scaleLinear()
      .domain([yExtent[0] - yPadding, yExtent[1] + yPadding])
      .range([this.height - this.margin.bottom, this.margin.top]);

    // Color scale for clusters
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

    // Draw points
    plotArea
      .selectAll("circle")
      .data(this.data.points)
      .enter()
      .append("circle")
      .attr("cx", (d) => xScale(d.x))
      .attr("cy", (d) => yScale(d.y))
      .attr("r", 5)
      .attr("fill", (d) => {
        return d.cluster === -1 ? "#ccc" : colorScale(d.cluster);
      })
      .attr("stroke", "#fff")
      .attr("stroke-width", 1)
      .on("mouseover", (event, d) => this.showTooltip(event, d))
      .on("mouseout", () => this.hideTooltip());

    // Add legend
    this.addLegend(colorScale);
  }

  // Show tooltip with data
  showTooltip(event, d) {
    // Get all properties except x, y, and cluster
    const properties = Object.keys(d).filter(
      (key) => !["x", "y", "cluster"].includes(key)
    );

    // Create tooltip content
    let content = `<strong>Cluster:</strong> ${
      d.cluster === -1 ? "Noise" : d.cluster
    }<br/>`;

    // Add all other properties
    properties.forEach((prop) => {
      content += `<strong>${prop}:</strong> ${d[prop]}<br/>`;
    });

    // Position and show tooltip
    this.tooltip
      .html(content)
      .style("left", event.pageX + 10 + "px")
      .style("top", event.pageY - 28 + "px")
      .transition()
      .duration(200)
      .style("opacity", 0.9);
  }

  // Hide tooltip
  hideTooltip() {
    this.tooltip.transition().duration(500).style("opacity", 0);
  }

  // Add legend for clusters
  addLegend(colorScale) {
    // Remove any existing legend
    d3.select(this.container).selectAll(".vis-legend").remove();
    
    const legend = d3
      .select(this.container)
      .append("div")
      .attr("class", "vis-legend");

    const legendTitle = legend
      .append("div")
      .attr("class", "legend-title")
      .text("Clusters");

    const clusters = this.data.clusters;
    
    clusters.forEach((cluster) => {
      const item = legend.append("div").attr("class", "legend-item");
      
      item
        .append("span")
        .attr("class", "legend-color")
        .style("background-color", cluster === -1 ? "#ccc" : colorScale(cluster));
      
      item
        .append("span")
        .attr("class", "legend-label")
        .text(cluster === -1 ? "Noise" : `Cluster ${cluster}`);
    });
  }
}

// Initialize visualization with data
function initVisualization(data) {
  const visualizer = new EmbeddingsVisualizer("visualization");
  visualizer.initialize(data);
} 