import React, { useState, useEffect } from 'react';
import { Search, Link2, AlertTriangle, TrendingUp, Activity, Globe, ChevronRight, Zap, Shield, Newspaper, Eye } from 'lucide-react';

export default function PhylosUI() {
  const [url, setUrl] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [graphData, setGraphData] = useState(null);
  const [selectedNode, setSelectedNode] = useState(null);
  const [mutationEvents, setMutationEvents] = useState([]);

  // Simulated WebSocket connection
  const startAnalysis = () => {
    if (!url) return;
    
    setIsAnalyzing(true);
    setGraphData(null);
    setMutationEvents([]);
    
    // Simulate receiving data
    setTimeout(() => {
      const mockData = {
        nodes: [
          { id: '1', title: 'Original Article', url: url, mutationScore: 0, timestamp: '2025-11-15 10:00', author: 'J. Doe' },
          { id: '2', title: 'First Republish', url: 'example.com/article-2', mutationScore: 0.12, timestamp: '2025-11-15 14:30', author: 'Daily Press' },
          { id: '3', title: 'Spin Version A', url: 'example.com/article-3', mutationScore: 0.67, timestamp: '2025-11-16 09:15', author: 'News Corp' },
          { id: '4', title: 'Spin Version B', url: 'example.com/article-4', mutationScore: 0.82, timestamp: '2025-11-16 11:45', author: 'Tribune' },
          { id: '5', title: 'Near Copy', url: 'example.com/article-5', mutationScore: 0.08, timestamp: '2025-11-17 08:20', author: 'Times' }
        ],
        edges: [
          { from: '1', to: '2' },
          { from: '1', to: '5' },
          { from: '2', to: '3' },
          { from: '3', to: '4' }
        ]
      };
      
      setGraphData(mockData);
      setMutationEvents([
        { node: '3', score: 0.67, type: 'semantic_drift', timestamp: '2025-11-16 09:15' },
        { node: '4', score: 0.82, type: 'high_mutation', timestamp: '2025-11-16 11:45' }
      ]);
      setIsAnalyzing(false);
    }, 2000);
  };

  const getMutationColor = (score) => {
    if (score < 0.2) return 'bg-emerald-800';
    if (score < 0.5) return 'bg-amber-700';
    return 'bg-red-900';
  };

  const getMutationLabel = (score) => {
    if (score < 0.2) return 'Authentic';
    if (score < 0.5) return 'Modified';
    return 'Corrupted';
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-blue-950 to-indigo-950 text-gray-100">
      {/* Newspaper texture overlay */}
      <div className="fixed inset-0 opacity-5 pointer-events-none" style={{
        backgroundImage: `url("data:image/svg+xml,%3Csvg width='100' height='100' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' /%3E%3C/filter%3E%3Crect width='100' height='100' filter='url(%23noise)' opacity='0.4'/%3E%3C/svg%3E")`
      }} />

      {/* Header with old newspaper masthead style */}
      <div className="border-b-4 border-amber-800/30 bg-black/40 backdrop-blur-sm relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-amber-900/5 to-transparent" />
        <div className="max-w-7xl mx-auto px-6 py-6 relative">
          <div className="text-center border-b border-amber-800/20 pb-4 mb-4">
            <div className="flex items-center justify-center gap-3 mb-2">
              <div className="h-px w-20 bg-gradient-to-r from-transparent to-amber-700/50" />
              <Newspaper className="w-5 h-5 text-amber-700/70" />
              <div className="h-px w-20 bg-gradient-to-l from-transparent to-amber-700/50" />
            </div>
            <h1 className="text-6xl mb-2 text-amber-100/90 tracking-wide" style={{ 
              fontFamily: '"UnifrakturMaguntia", "Old English Text MT", "Blackletter", serif',
              textShadow: '2px 2px 4px rgba(0,0,0,0.7)',
              fontWeight: 400
            }}>
              Phylos
            </h1>
            <style>{`
              @import url('https://fonts.googleapis.com/css2?family=UnifrakturMaguntia&display=swap');
            `}</style>
            <div className="flex items-center justify-center gap-2 text-xs text-amber-700/70 uppercase tracking-widest">
              <div className="h-px w-12 bg-amber-800/30" />
              <span>Narrative Evolution Chronicle</span>
              <div className="h-px w-12 bg-amber-800/30" />
            </div>
          </div>
          <p className="text-center text-sm text-amber-200/60 italic font-serif">
            "Tracing the Lineage of Truth Through the Shadows of Disinformation"
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-6 py-8 relative">
        {/* Search Section - styled like a classified ad box */}
        <div className="mb-8">
          <div className="bg-gradient-to-br from-slate-800/60 to-slate-900/80 rounded-sm p-8 border-4 border-double border-amber-900/40 backdrop-blur-sm shadow-2xl relative">
            <div className="absolute top-4 left-4 text-6xl text-amber-900/10 font-serif">"</div>
            <div className="absolute bottom-4 right-4 text-6xl text-amber-900/10 font-serif">"</div>
            
            <div className="flex items-center gap-3 mb-4">
              <Eye className="w-5 h-5 text-amber-700" />
              <h2 className="text-xl font-serif text-amber-100/90">Investigation Desk</h2>
            </div>
            <p className="text-amber-200/60 mb-6 text-sm font-serif italic">
              Submit a source URL to trace its narrative mutations across the information network
            </p>
            
            <div className="flex gap-3">
              <div className="flex-1 relative">
                <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-amber-700/50" />
                <input
                  type="text"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://source-article.com/original-story"
                  className="w-full bg-black/60 border-2 border-amber-900/40 rounded-sm px-12 py-4 text-amber-100 placeholder-amber-700/40 focus:outline-none focus:border-amber-700/60 focus:ring-2 focus:ring-amber-700/20 font-mono text-sm"
                  onKeyPress={(e) => e.key === 'Enter' && startAnalysis()}
                />
              </div>
              <button
                onClick={startAnalysis}
                disabled={isAnalyzing || !url}
                className="px-8 py-4 bg-gradient-to-b from-amber-800 to-amber-900 rounded-sm font-serif uppercase text-sm tracking-wider hover:from-amber-700 hover:to-amber-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2 shadow-lg border border-amber-700/50"
              >
                {isAnalyzing ? (
                  <>
                    <div className="w-5 h-5 border-2 border-amber-200/30 border-t-amber-200 rounded-full animate-spin" />
                    Investigating
                  </>
                ) : (
                  <>
                    <Zap className="w-5 h-5" />
                    Trace Origin
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Results Grid */}
        {graphData && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Graph Visualization - styled like newspaper columns */}
            <div className="lg:col-span-2 bg-gradient-to-br from-slate-900/80 to-slate-800/60 rounded-sm p-6 border-2 border-amber-900/30 backdrop-blur-sm shadow-2xl">
              <div className="border-b-2 border-amber-900/30 pb-4 mb-6">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <Link2 className="w-5 h-5 text-amber-700" />
                    <h3 className="text-xl font-serif text-amber-100/90">Citation Network</h3>
                  </div>
                  <div className="flex gap-3 text-xs font-serif">
                    <div className="flex items-center gap-1.5">
                      <div className="w-3 h-3 bg-emerald-800 border border-emerald-600" />
                      <span className="text-emerald-300">Authentic</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="w-3 h-3 bg-amber-700 border border-amber-500" />
                      <span className="text-amber-300">Modified</span>
                    </div>
                    <div className="flex items-center gap-1.5">
                      <div className="w-3 h-3 bg-red-900 border border-red-700" />
                      <span className="text-red-300">Corrupted</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Newspaper-style article cards */}
              <div className="space-y-4">
                {graphData.nodes.map((node, idx) => (
                  <div
                    key={node.id}
                    onClick={() => setSelectedNode(node)}
                    className={`p-5 border-2 cursor-pointer transition-all ${
                      selectedNode?.id === node.id
                        ? 'bg-amber-900/20 border-amber-700/60 shadow-lg'
                        : 'bg-black/30 border-amber-900/30 hover:border-amber-800/50 hover:bg-amber-950/20'
                    }`}
                  >
                    <div className="flex items-start gap-4">
                      <div className={`w-14 h-14 ${getMutationColor(node.mutationScore)} border-2 ${
                        node.mutationScore < 0.2 ? 'border-emerald-600' :
                        node.mutationScore < 0.5 ? 'border-amber-500' :
                        'border-red-700'
                      } flex items-center justify-center flex-shrink-0 shadow-lg`}>
                        {node.mutationScore < 0.2 ? (
                          <Shield className="w-7 h-7 text-emerald-100" />
                        ) : node.mutationScore < 0.5 ? (
                          <TrendingUp className="w-7 h-7 text-amber-100" />
                        ) : (
                          <AlertTriangle className="w-7 h-7 text-red-100" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex-1">
                            <h4 className="font-serif text-lg font-bold text-amber-50 mb-1">{node.title}</h4>
                            <p className="text-xs text-amber-700/70 uppercase tracking-wide font-serif">By {node.author}</p>
                          </div>
                          <span className={`text-xs px-3 py-1 border-2 ml-4 ${
                            node.mutationScore < 0.2 ? 'bg-emerald-900/40 border-emerald-700 text-emerald-200' :
                            node.mutationScore < 0.5 ? 'bg-amber-900/40 border-amber-700 text-amber-200' :
                            'bg-red-900/40 border-red-700 text-red-200'
                          } font-serif uppercase tracking-wider`}>
                            {getMutationLabel(node.mutationScore)}
                          </span>
                        </div>
                        <p className="text-sm text-amber-300/50 font-mono truncate border-l-2 border-amber-900/30 pl-3">{node.url}</p>
                        <div className="flex items-center gap-4 mt-3 text-xs text-amber-600/60 font-serif italic border-t border-amber-900/20 pt-2">
                          <span>Mutation Index: {node.mutationScore.toFixed(3)}</span>
                          <span>•</span>
                          <span>Published: {node.timestamp}</span>
                        </div>
                      </div>
                      <ChevronRight className="w-5 h-5 text-amber-700/40" />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Sidebar - styled like newspaper sidebars */}
            <div className="space-y-6">
              {/* Stats Box */}
              <div className="bg-gradient-to-br from-slate-900/80 to-slate-800/60 rounded-sm p-6 border-2 border-amber-900/30 backdrop-blur-sm shadow-2xl">
                <h3 className="text-lg font-serif text-amber-100/90 mb-4 pb-2 border-b-2 border-amber-900/30 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-amber-700" />
                  Investigation Report
                </h3>
                <div className="space-y-4 font-serif">
                  <div className="flex justify-between items-center border-b border-amber-900/20 pb-2">
                    <span className="text-amber-300/70 text-sm">Sources Traced</span>
                    <span className="font-bold text-2xl text-amber-100">{graphData.nodes.length}</span>
                  </div>
                  <div className="flex justify-between items-center border-b border-amber-900/20 pb-2">
                    <span className="text-amber-300/70 text-sm">Corruptions Found</span>
                    <span className="font-bold text-2xl text-red-300">{mutationEvents.length}</span>
                  </div>
                  <div className="flex justify-between items-center border-b border-amber-900/20 pb-2">
                    <span className="text-amber-300/70 text-sm">Avg. Mutation</span>
                    <span className="font-bold text-2xl text-amber-100">
                      {(graphData.nodes.reduce((a, b) => a + b.mutationScore, 0) / graphData.nodes.length).toFixed(3)}
                    </span>
                  </div>
                </div>
              </div>

              {/* Mutation Alert Box */}
              <div className="bg-gradient-to-br from-red-950/60 to-slate-900/80 rounded-sm p-6 border-2 border-red-900/40 backdrop-blur-sm shadow-2xl">
                <h3 className="text-lg font-serif text-red-200 mb-4 pb-2 border-b-2 border-red-900/30 flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-red-400" />
                  Corruption Alerts
                </h3>
                <div className="space-y-3">
                  {mutationEvents.map((event, idx) => (
                    <div key={idx} className="p-4 bg-black/40 border-2 border-red-900/30">
                      <div className="flex items-center justify-between mb-2 pb-2 border-b border-red-900/20">
                        <span className="text-xs font-serif text-red-300 uppercase tracking-wider">
                          {event.type.replace('_', ' ')}
                        </span>
                        <span className="text-sm text-red-200 font-bold font-mono">
                          {event.score.toFixed(3)}
                        </span>
                      </div>
                      <p className="text-xs text-amber-400/60 font-serif italic">{event.timestamp}</p>
                    </div>
                  ))}
                  {mutationEvents.length === 0 && (
                    <p className="text-sm text-red-300/50 font-serif italic text-center py-4">
                      No major corruptions detected
                    </p>
                  )}
                </div>
              </div>

              {/* Selected Node Details */}
              {selectedNode && (
                <div className="bg-gradient-to-br from-slate-900/80 to-slate-800/60 rounded-sm p-6 border-2 border-amber-900/30 backdrop-blur-sm shadow-2xl">
                  <h3 className="text-lg font-serif text-amber-100/90 mb-4 pb-2 border-b-2 border-amber-900/30">
                    Article Details
                  </h3>
                  <div className="space-y-4 text-sm font-serif">
                    <div>
                      <span className="text-amber-600/80 block mb-1 uppercase text-xs tracking-wide">Headline</span>
                      <span className="font-medium text-amber-100">{selectedNode.title}</span>
                    </div>
                    <div>
                      <span className="text-amber-600/80 block mb-1 uppercase text-xs tracking-wide">Source URL</span>
                      <span className="font-mono text-xs text-amber-300/80 break-all">{selectedNode.url}</span>
                    </div>
                    <div>
                      <span className="text-amber-600/80 block mb-1 uppercase text-xs tracking-wide">Author</span>
                      <span className="text-amber-200/90">{selectedNode.author}</span>
                    </div>
                    <div>
                      <span className="text-amber-600/80 block mb-2 uppercase text-xs tracking-wide">Mutation Index</span>
                      <div className="flex items-center gap-3">
                        <div className="flex-1 h-3 bg-black/40 border border-amber-900/30 overflow-hidden">
                          <div
                            className={`h-full ${getMutationColor(selectedNode.mutationScore)} border-r-2 ${
                              selectedNode.mutationScore < 0.2 ? 'border-emerald-500' :
                              selectedNode.mutationScore < 0.5 ? 'border-amber-500' :
                              'border-red-500'
                            }`}
                            style={{ width: `${selectedNode.mutationScore * 100}%` }}
                          />
                        </div>
                        <span className="font-bold font-mono text-amber-100">{selectedNode.mutationScore.toFixed(3)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Empty State - styled like a missing persons ad */}
        {!graphData && !isAnalyzing && (
          <div className="text-center py-20">
            <div className="w-32 h-32 bg-amber-950/20 border-4 border-double border-amber-900/40 flex items-center justify-center mx-auto mb-6">
              <Newspaper className="w-16 h-16 text-amber-700/40" />
            </div>
            <h3 className="text-2xl font-serif text-amber-200/70 mb-3">
              Awaiting Investigation
            </h3>
            <p className="text-amber-400/50 text-sm font-serif italic max-w-md mx-auto">
              Submit a source article above to begin tracing its narrative evolution through the shadows of the information age
            </p>
            <div className="mt-6 flex items-center justify-center gap-2">
              <div className="h-px w-20 bg-gradient-to-r from-transparent to-amber-800/30" />
              <span className="text-xs text-amber-700/50 font-serif">◆</span>
              <div className="h-px w-20 bg-gradient-to-l from-transparent to-amber-800/30" />
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="border-t-2 border-amber-900/30 bg-black/40 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-4 text-center">
          <p className="text-xs text-amber-700/60 font-serif italic">
            Est. 2025 • Phylos Chronicle • Shedding Light on Digital Shadows
          </p>
        </div>
      </div>
    </div>
  );
}