const inputCls = "w-full px-4 py-3 rounded-xl border border-slate-200 bg-slate-50/50 text-sm text-slate-800 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-medical-500/20 focus:border-medical-400 transition-all"

export default function MedicalInputField({
  label,
  value,
  onChange,
  type = 'number',
  placeholder = '',
  min,
  max,
  step,
  icon: Icon,
  iconColor = 'text-slate-400',
  required = false,
}) {
  const handleChange = (e) => {
    const val = e.target.value
    onChange(val)
  }

  return (
    <div className="space-y-1.5">
      <label className="block text-sm font-medium text-slate-600 mb-1.5 flex items-center gap-1.5">
        {Icon && <Icon className={`w-3.5 h-3.5 ${iconColor}`} />}
        {label}
        {required && <span className="text-critical-400">*</span>}
      </label>
      <input
        type={type}
        value={value ?? ''}
        onChange={handleChange}
        placeholder={placeholder}
        min={min}
        max={max}
        step={step}
        required={required}
        className={inputCls}
      />
    </div>
  )
}
