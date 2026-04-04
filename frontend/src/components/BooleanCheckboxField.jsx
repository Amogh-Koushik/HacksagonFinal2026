export default function BooleanCheckboxField({
  label,
  value,
  onChange,
  icon: Icon,
  iconColor = 'text-slate-400',
}) {
  const isChecked = value === 1 || value === true

  const handleToggle = () => {
    const newValue = !isChecked
    onChange(newValue ? 1 : 0)
  }

  return (
    <div className="min-h-[40px]">
      <button
        type="button"
        onClick={handleToggle}
        className="w-full flex items-center justify-between gap-3 cursor-pointer select-none group text-left"
        aria-pressed={isChecked}
        aria-label={label}
      >
        <div className="flex items-center gap-2.5 min-w-0">
          <div className={`w-5 h-5 shrink-0 rounded-md border-2 flex items-center justify-center transition-all ${
          isChecked
            ? 'bg-medical-500 border-medical-500 shadow-md shadow-medical-500/30'
            : 'border-slate-300 group-hover:border-medical-400'
        }`}>
            {isChecked && (
              <svg className="w-3 h-3 text-white" viewBox="0 0 12 12" fill="none">
                <path d="M2 6L5 9L10 3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            )}
          </div>
          <div className="flex items-center gap-1.5 min-w-0">
            {Icon && <Icon className={`w-3.5 h-3.5 ${iconColor}`} />}
            <span className="text-sm font-medium text-slate-700 truncate">{label}</span>
          </div>
        </div>
        <span className={`text-[11px] font-semibold px-2 py-0.5 rounded-md shrink-0 ${
          isChecked ? 'bg-medical-50 text-medical-700' : 'bg-slate-100 text-slate-500'
        }`}>
          {isChecked ? 'YES' : 'NO'}
        </span>
      </button>
    </div>
  )
}
