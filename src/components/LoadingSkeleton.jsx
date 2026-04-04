import { motion } from 'framer-motion'

export function CardSkeleton() {
  return (
    <div className="bg-white rounded-2xl border border-slate-200/60 p-6 animate-pulse">
      <div className="flex items-start justify-between">
        <div className="space-y-3 flex-1">
          <div className="h-3 w-24 bg-slate-200 rounded-full" />
          <div className="h-8 w-16 bg-slate-200 rounded-lg" />
          <div className="h-2.5 w-20 bg-slate-100 rounded-full" />
        </div>
        <div className="w-12 h-12 bg-slate-100 rounded-xl" />
      </div>
    </div>
  )
}

export function TableSkeleton({ rows = 5 }) {
  return (
    <div className="bg-white rounded-2xl border border-slate-200/60 overflow-hidden animate-pulse">
      <div className="px-6 py-4 border-b border-slate-100">
        <div className="h-4 w-32 bg-slate-200 rounded-full" />
      </div>
      {Array.from({ length: rows }).map((_, i) => (
        <div key={i} className="px-6 py-4 border-b border-slate-50 flex items-center gap-4">
          <div className="w-8 h-8 bg-slate-100 rounded-lg" />
          <div className="flex-1 space-y-2">
            <div className="h-3 w-32 bg-slate-100 rounded-full" />
            <div className="h-2.5 w-48 bg-slate-50 rounded-full" />
          </div>
          <div className="h-6 w-16 bg-slate-100 rounded-full" />
          <div className="h-3 w-12 bg-slate-50 rounded-full" />
        </div>
      ))}
    </div>
  )
}

export function PageSkeleton() {
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-6">
      <div className="space-y-2">
        <div className="h-6 w-48 bg-slate-200 rounded-lg animate-pulse" />
        <div className="h-3 w-72 bg-slate-100 rounded-full animate-pulse" />
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Array.from({ length: 4 }).map((_, i) => <CardSkeleton key={i} />)}
      </div>
      <TableSkeleton />
    </motion.div>
  )
}
