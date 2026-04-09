function Header(el)
    if el.level == 4 then
      local name = pandoc.utils.stringify(el.content[3])
      local number = ""
      if el.attr then
        number = el.attr.attributes.number:gsub("%.0%.0%.", "-")
      end
      el.content =  pandoc.Strong(name .. " " .. number)
      return el
    end
  end
  