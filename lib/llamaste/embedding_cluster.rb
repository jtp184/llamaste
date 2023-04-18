# frozen_string_literal: true

module Llamaste
  # K-means clustering for embeddings
  class EmbeddingCluster
    # The set of TextEmbeddings
    attr_reader :embeddings
    # The number of clusters
    attr_reader :k
    # The clustered data
    attr_reader :clusters

    # Takes in an array of TextEmbeddings and the number of +k_clusters+ to group by
    def initialize(embeddings, k_clusters)
      @embeddings = embeddings
      @k = k_clusters
      @clusters = Array.new(k) { [] }
      @centroids = initial_centroids
    end

    # Uses the +limit+ for max iterations, and clusters the data
    def call(limit = 100)
      limit.times do
        assign_clusters
        new_centroids = update_centroids

        break if new_centroids == centroids

        self.centroids = new_centroids
        clusters.map!(&:clear)
      end

      clusters
    end

    # Takes in the +embeddings+, number of +k_clusters+ and optional +limit+ and returns a calculated cluster
    def self.call(embeddings, k_clusters, limit = 100)
      new(embeddings, k_clusters).tap { |c| c.call(limit) }
    end

    private

    # The cluster anchor points
    attr_accessor :centroids

    # Creates the randomized centroids
    def initial_centroids
      embeddings.sample(@k).map(&:vector)
    end

    # Fits the data using the minimal distance for each embedding
    def assign_clusters
      embeddings.each do |embedding|
        closest = centroids.each_with_index.min_by do |centroid, _|
          distance(embedding.vector, centroid)
        end.last

        clusters[closest] << embedding
      end
    end

    # Returns the summated value of each cluster
    def update_centroids
      clusters.map do |cluster|
        next Matrix[[0] * embeddings.first.vector.column_count] if cluster.empty?

        sum = cluster.map(&:vector).reduce(:+)
        sum / cluster.size
      end
    end

    # Gets the norm distance between the +left+ and +right+ vectors
    def distance(left, right)
      (left - right).column_vectors.first.norm
    end
  end
end
