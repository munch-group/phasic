function Header(el)
  if el.level == 4 then
    el.classes:insert("unnumbered")
    return el
  end
end
